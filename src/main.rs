use anyhow::{bail, ensure, Context as _};
use gumdrop::Options;
use polyfuse::{
    io::{Reader, Writer},
    op,
    reply::{ReplyAttr, ReplyEntry},
    Context, DirEntry, FileAttr, Filesystem, Operation,
};
use serde::{Deserialize, Serialize};
#[cfg(doc)]
#[doc(inline)]
pub use std;
use std::convert::From;
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{self, prelude::*};
use std::path::{Path, PathBuf};
use std::{borrow::Cow, time::Duration};
use thiserror::Error;
use tracing::trace;
use tracing_futures::Instrument;

const TTL: Duration = Duration::from_secs(60 * 60 * 24 * 365);
const ROOT_INO: INodeId = INodeId(1);

#[derive(Eq, PartialEq, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct INodeId(u64);

impl Display for INodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl INodeId {
    fn u64(self) -> u64 {
        self.0
    }
    fn usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u64> for INodeId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<usize> for INodeId {
    fn from(v: usize) -> Self {
        (v as u64).into()
    }
}

#[derive(Debug, Error)]
pub enum DebError {
    #[error("ar file missing a required entry {0:?}")]
    MissingArEntry(ArEntry),
    #[error("dir and non-dir cannot both exist at path {0:?}")]
    KindMismatch(PathBuf),
    #[error("unsupported deb file version {0:?}")]
    UnsupportedVersion(String),
}

#[derive(Debug, Error)]
pub enum FSError {
    #[error("is not dir")]
    IsNotDir,
    #[error("path {0} not found")]
    NotFound(PathBuf),
    #[error("dangling reference to inode {0} found: {1}")]
    DanglingReference(INodeId, String),
    #[error("{0}")]
    Other(anyhow::Error),
}

type FSResult<T> = Result<T, FSError>;

impl From<FSError> for io::Error {
    fn from(e: FSError) -> Self {
        Self::new(io::ErrorKind::Other, e)
    }
}
impl From<anyhow::Error> for FSError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

#[derive(Debug)]
pub enum ArEntry {
    Version,
    Control,
    Data,
}

#[derive(Debug, Serialize, Deserialize)]
struct INode {
    id: INodeId,
    parent: INodeId,
    name: OsString,
    mode: u32,
    content: INodeContent,
    prev: Option<Box<INode>>,
}

impl INode {
    pub fn get_children(&self) -> FSResult<&[INodeId]> {
        if let INodeContent::Dir(ref children) = self.content {
            Ok(children)
        } else {
            Err(FSError::IsNotDir.into())
        }
    }
    pub fn get_mut_children(&mut self) -> Result<&mut Vec<INodeId>, FSError> {
        if let INodeContent::Dir(ref mut children) = self.content {
            Ok(children)
        } else {
            Err(FSError::IsNotDir.into())
        }
    }
    pub fn is(&self, kind: INodeKind) -> bool {
        self.kind() == kind
    }
    pub fn is_dir(&self) -> bool {
        self.is(INodeKind::Dir)
    }
    fn attr(&self) -> FileAttr {
        let mut attr = FileAttr::default();
        attr.set_mode(
            if self.is_dir() {
                libc::S_IFDIR
            } else {
                libc::S_IFREG
            } as u32
                | self.mode,
        );
        attr.set_ino(self.id.u64());
        attr.set_nlink(if self.is_dir() { 2 } else { 1 });
        attr.set_uid(unsafe { libc::getuid() });
        attr.set_gid(unsafe { libc::getgid() });
        if let INodeContent::File { deb: _, size } = self.content {
            attr.set_size(size)
        }
        attr
    }
    fn kind(&self) -> INodeKind {
        self.content.kind()
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum INodeContent {
    File { deb: usize, size: u64 },
    Dir(Vec<INodeId>),
    Symlink(PathBuf),
}

impl INodeContent {
    fn kind(&self) -> INodeKind {
        match self {
            Self::File { .. } => INodeKind::File,
            Self::Dir(_) => INodeKind::Dir,
            Self::Symlink(_) => INodeKind::Symlink,
        }
    }
}

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
enum INodeKind {
    File,
    Dir,
    Symlink,
}

#[derive(Options)]
struct Opts {
    #[options(free)]
    debs: Vec<PathBuf>,
    mount_point: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let opts = Opts::parse_args_default_or_exit();
    ensure!(
        opts.mount_point.is_dir(),
        "the mountpoint must be a directory"
    );
    let savfile = opts.mount_point.join(".debfs");
    let mut fs = if savfile.is_file() {
        serde_json::from_reader(File::open(&savfile)?)?
    } else {
        DebFS::new()
    };
    for deb in opts.debs {
        if fs.add_deb(deb)? {
            serde_json::to_writer_pretty(File::create(&savfile)?, &fs)?;
        }
    }
    polyfuse_tokio::mount(
        fs,
        opts.mount_point,
        &[OsStr::new("-o"), OsStr::new("nonempty")],
    )
    .await?;
    Ok(())
}

fn check_deb_version<R: Read + Seek>(entry: ar::Entry<R>) -> anyhow::Result<()> {
    // TODO: Return Result<(), DebError>
    ensure!(
        entry.header().identifier() == b"debian-binary",
        format!(
            "expected 'debian-binary' but got '{:?}'",
            entry.header().identifier()
        )
    );
    let version_str = io::BufReader::with_capacity(8, entry)
        .lines()
        .next()
        .context("deb file version is missing")??;

    if version_str.starts_with("2.") {
        Ok(())
    } else {
        Err(DebError::UnsupportedVersion(version_str).into())
    }
}

#[derive(Serialize, Deserialize)]
struct DebFS {
    inodes: Vec<Option<Box<INode>>>,
    debs: Vec<PathBuf>,
    //    handles:
}

fn with_deb<
    P: AsRef<Path>,
    CF: FnOnce(tar::Archive<&mut (dyn Read + Send)>) -> anyhow::Result<CR>,
    DF: FnOnce(tar::Archive<&mut (dyn Read + Send)>) -> anyhow::Result<DR>,
    CR,
    DR,
>(
    path: P,
    control_fn: CF,
    data_fn: DF,
) -> anyhow::Result<(CR, DR)> {
    let path = path.as_ref().to_path_buf();
    let mut archive = ar::Archive::new(File::open(&path)?);
    check_deb_version(
        archive
            .next_entry()
            .ok_or(DebError::MissingArEntry(ArEntry::Version))
            .with_context(|| format!("while parsing version entry in file {}", path.display()))??,
    )
    .with_context(|| format!("in file {}", path.display()))?;
    let cr = {
        let mut cr_entry = archive
            .next_entry()
            .ok_or(DebError::MissingArEntry(ArEntry::Control))??;
        with_tarball(&mut cr_entry, b"control", control_fn)?
    };
    let mut data_entry = archive
        .next_entry()
        .ok_or(DebError::MissingArEntry(ArEntry::Data))??;
    let dr = with_tarball(&mut data_entry, b"data", data_fn)?;
    Ok((cr, dr))
}

fn with_tarball<R, F: FnOnce(tar::Archive<&mut (dyn Read + Send)>) -> anyhow::Result<R>>(
    mut data_entry: &mut ar::Entry<File>,
    prefix: &'static [u8],
    fun: F,
) -> anyhow::Result<R> {
    let name = data_entry.header().identifier();
    if !name.starts_with(prefix) {
        bail!(
            "Expected entry name to start with {:?} but got {} ({:?})",
            String::from_utf8_lossy(prefix),
            String::from_utf8_lossy(name),
            name
        );
    }
    let mut reader: Box<dyn Read + Send> = match &name[prefix.len()..] {
        b".tar.xz" | b".tar.7z" => Box::new(lzma::LzmaReader::new_decompressor(data_entry)?),
        b".tar" => Box::new(&mut data_entry),
        _ => {
            bail!(
                "Unrecognized data tar filename {} ({:?})",
                String::from_utf8_lossy(name),
                name
            );
        }
    };
    fun(tar::Archive::new(reader.as_mut()))
}

impl DebFS {
    fn new() -> Self {
        DebFS {
            inodes: vec![
                None,
                Some(Box::new(INode {
                    id: ROOT_INO,
                    parent: ROOT_INO,
                    name: ".".to_owned().into(),
                    mode: 0o644,
                    content: INodeContent::Dir(Vec::new()),
                    prev: None,
                })),
            ],
            debs: Vec::new(),
        }
    }

    fn add_deb<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<bool> {
        let canonical_path = path.as_ref().canonicalize()?;
        if self.debs.contains(&canonical_path) {
            return Ok(false);
        }
        with_deb(path, |_| Ok(()), |tb| self._add_tarball(canonical_path, tb))?;
        Ok(true)
    }

    fn _add_tarball(
        &mut self,
        path: PathBuf,
        mut tarball: tar::Archive<&mut (dyn Read + Send)>,
    ) -> anyhow::Result<()> {
        let deb = self.debs.len();
        self.debs.push(path);
        for entry in tarball.entries()? {
            let entry = entry?;
            let path: Cow<Path> = entry.header().path()?;
            trace!("{} => ", path.display());
            let path = if path.ends_with("/") {
                if let Some(parent) = path.parent() {
                    parent.into()
                } else {
                    continue;
                }
            } else {
                path
            };
            let path = if path.starts_with("./") {
                path.strip_prefix("./")?.into()
            } else {
                path
            };
            trace!("{}: {:?}", path.display(), entry.header().entry_type());
            let dir_inode = if let Some(parent) = path.parent() {
                self.lookup(parent)?
                    .with_context(|| format!("parent dir does not exist while adding {:?}", path))?
            } else {
                ROOT_INO
            };
            let name = path.file_name().unwrap_or(OsStr::new("")).to_os_string();
            if name == OsStr::new("") || name == OsStr::new(".") {
                continue;
            }
            let mut inode = Box::new(INode {
                id: INodeId(0),
                parent: dir_inode,
                name: name,
                mode: entry.header().mode()?,
                content: match entry.header().entry_type() {
                    tar::EntryType::Regular => INodeContent::File {
                        deb: deb,
                        size: entry.header().size()?,
                    },
                    tar::EntryType::Directory => INodeContent::Dir(Vec::new()),
                    tar::EntryType::Symlink => INodeContent::Symlink(
                        entry
                            .link_name()?
                            .with_context(|| {
                                format!("error reading link name for path {:?}", path)
                            })?
                            .to_path_buf(),
                    ),
                    t => {
                        trace!("Unsupported entry type: {:?}", t);
                        continue;
                    }
                },
                prev: None,
            });
            if let Some(prev_id) = self.lookup1(dir_inode, &inode.name)? {
                if inode.is_dir()
                    != self
                        .get(prev_id, || format!("while replacing path {:?}", path))?
                        .is_dir()
                {
                    return Err(DebError::KindMismatch(path.to_path_buf()).into());
                } else if inode.is_dir() {
                    continue;
                }
                inode.id = prev_id;
                inode.prev = self.inodes[prev_id.usize()].take();
                self.inodes[prev_id.usize()] = Some(inode)
            } else {
                let id = self.inodes.len().into();
                inode.id = id;
                self.inodes.push(Some(inode));
                let parent_path = self.path(dir_inode);
                let dir = self.inodes[dir_inode.usize()].as_mut().unwrap();
                dir.get_mut_children()
                    .with_context(|| {
                        format!("path {} (inode {})", parent_path.display(), dir_inode)
                    })?
                    .push(id);
            }
        }

        Ok(())
    }

    fn path(&self, mut inode: INodeId) -> PathBuf {
        let mut parts = Vec::new();
        while inode != ROOT_INO {
            let ino = self.inodes[inode.usize()].as_ref().unwrap();
            parts.push(ino.name.clone());
            inode = ino.parent;
        }
        parts.iter().rev().collect()
    }

    fn get<EF: FnOnce() -> String>(&self, inode: INodeId, error_func: EF) -> FSResult<&INode> {
        self.inodes
            .get(inode.usize())
            .map(Option::as_ref)
            .flatten()
            .ok_or_else(|| FSError::DanglingReference(inode, error_func()))
            .map(Box::as_ref)
    }

    fn lookup<P: AsRef<Path>>(&self, path: P) -> FSResult<Option<INodeId>> {
        self.lookup_in(ROOT_INO, path.as_ref())
    }

    fn lookup_in(&self, inode: INodeId, path: &Path) -> FSResult<Option<INodeId>> {
        trace!("{} => ", path.display());
        let path = if path.ends_with("/") {
            path.parent()
                .with_context(|| format!("unable to strip trailing slash from {}", path.display()))?
                .into()
        } else {
            path
        };
        let path = if path.starts_with("./") {
            path.strip_prefix("./")
                .with_context(|| format!("failed to strip ./ prefix from {:?}", path))?
                .into()
        } else {
            path
        };
        trace!("{}", path.display());
        let mut cur_inode = inode;
        for name in path.iter() {
            if name == OsStr::new("") || name == OsStr::new(".") {
                continue;
            }
            cur_inode = if let Some(child_inode) = self.lookup1(cur_inode, name)? {
                child_inode
            } else {
                return Ok(None);
            };
            trace!(
                "{:?} -> {} ({}); ",
                name,
                self.path(cur_inode).display(),
                cur_inode
            );
        }
        Ok(Some(cur_inode))
    }

    fn lookup1(&self, inode: INodeId, name: &OsStr) -> Result<Option<INodeId>, FSError> {
        let err_fn = || format!("in call to lookup1({}, {:?})", inode, name);

        self.get(inode, err_fn)?
            .get_children()?
            .iter()
            .map(|&child| self.get(child, err_fn))
            .filter_map(|child_result| match child_result {
                Ok(c) => {
                    if c.name == name {
                        Some(Ok(c.id))
                    } else {
                        None
                    }
                }
                Err(e) => Some(Err(e)),
            })
            .next()
            .transpose()
    }

    async fn do_lookup(&self, op: &op::Lookup<'_>) -> io::Result<ReplyEntry>
where {
        let parent = op.parent().into();
        let name = op.name();
        let err_fn = || format!("in do_lookup() for inode {}, name {:?})", parent, name);
        let dir = self.get(parent, err_fn)?;
        if !dir.is_dir() {
            return Err(io::Error::from_raw_os_error(libc::ENOTDIR));
        }
        if let Some(inode) = self.lookup1(op.parent().into(), op.name())? {
            let inode = self.get(inode, err_fn)?;
            let mut reply = ReplyEntry::default();
            reply
                .ino(inode.id.u64())
                .attr(inode.attr())
                .ttl_attr(TTL)
                .ttl_entry(TTL);
            Ok(reply)
        } else {
            return Err(io::Error::from_raw_os_error(libc::ENOENT));
        }
    }
    async fn do_getattr(&self, op: &op::Getattr<'_>) -> io::Result<ReplyAttr>
where {
        if let Some(Some(inode)) = self.inodes.get(op.ino() as usize) {
            let mut reply = ReplyAttr::new(inode.attr());
            reply.ttl_attr(TTL);
            Ok(reply)
        } else {
            Err(io::Error::from_raw_os_error(libc::ENOENT))
        }
    }
    async fn do_read(&self, op: &op::Read<'_>) -> io::Result<Vec<u8>>
where {
        let inode = self
            .inodes
            .get(op.ino() as usize)
            .map(Option::as_ref)
            .flatten()
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))?;
        match inode.content {
            INodeContent::File { deb, .. } => {
                let path = self.path(inode.id);
                let deb_path = &self.debs[deb];
                trace!(
                    "reading {} bytes with offset {} from {} ({})...",
                    op.size(),
                    op.offset(),
                    path.display(),
                    deb_path.display()
                );
                let buf = with_deb(
                    deb_path,
                    |_| Ok(()),
                    |mut archive| {
                        let entry = archive
                            .entries()?
                            .filter_map(Result::ok)
                            .filter(|e| {
                                if let Ok(epath) = e.header().path() {
                                    let epath = if epath.ends_with("/") {
                                        if let Some(parent) = epath.parent() {
                                            parent.into()
                                        } else {
                                            return false;
                                        }
                                    } else {
                                        epath
                                    };
                                    let epath = if epath.starts_with("./") {
                                        if let Ok(epath) = epath.strip_prefix("./") {
                                            epath.into()
                                        } else {
                                            return false;
                                        }
                                    } else {
                                        epath
                                    };
                                    epath == path
                                } else {
                                    false
                                }
                            })
                            .next()
                            .with_context(|| {
                                format!("{:?} does not contain path {:?}", deb_path, path)
                            })?;
                        let offset = op.offset();
                        let size = op.size();
                        let mut err = Ok(());
                        let buf: Vec<u8> = entry
                            .bytes()
                            .skip(offset as usize)
                            .take(size as usize)
                            .filter_map(|e| match e {
                                Ok(b) => Some(b),
                                Err(er) => {
                                    if err.is_ok() {
                                        err = Err(er);
                                    }
                                    None
                                }
                            })
                            .collect();
                        err?;
                        Ok(buf)
                    },
                )
                .map(|(_, buf)| buf)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                trace!("read {} bytes", buf.len());
                Ok(buf)
            }
            _ => Err(io::Error::from_raw_os_error(libc::EINVAL)),
        }
    }
    async fn do_readdir(&self, op: &op::Readdir<'_>) -> io::Result<Vec<DirEntry>>
where {
        if let Some(Some(dir)) = self.inodes.get(op.ino() as usize) {
            if let INodeContent::Dir(ref children) = dir.content {
                let offset = op.offset() as usize;
                let size = op.size() as usize;

                let mut entries = Vec::with_capacity(children.len());
                let mut total_len: usize = 0;

                let default_entries = vec![
                    DirEntry::dir(".", dir.id.u64(), 1),
                    DirEntry::dir("..", dir.parent.u64(), 2),
                ];

                for entry in default_entries
                    .into_iter()
                    .chain(
                        children
                            .iter()
                            .copied()
                            .filter_map(|child| self.inodes[child.usize()].as_ref())
                            .enumerate()
                            .map(|(i, child)| (i as u64 + 3, child))
                            .map(|(i, child)| match child.kind() {
                                INodeKind::Dir => DirEntry::dir(&child.name, child.id.u64(), i),
                                INodeKind::File => DirEntry::file(&child.name, child.id.u64(), i),
                                INodeKind::Symlink => {
                                    dir_entry(libc::DT_LNK, &child.name, child.id.u64(), i)
                                }
                            }),
                    )
                    .skip(offset as usize)
                {
                    let entry_len = entry.as_ref().len();
                    if total_len + entry_len > size {
                        break;
                    }
                    entries.push(entry);
                    total_len += entry_len;
                }

                Ok(entries)
            } else {
                Err(io::Error::from_raw_os_error(libc::ENOTDIR))
            }
        } else {
            Err(io::Error::from_raw_os_error(libc::ENOENT))
        }
    }
    async fn do_readlink(&self, op: &op::Readlink<'_>) -> io::Result<PathBuf>
where {
        if let Some(Some(link)) = self.inodes.get(op.ino() as usize) {
            if let INodeContent::Symlink(ref target) = link.content {
                Ok(target.clone())
            } else {
                Err(io::Error::from_raw_os_error(libc::EINVAL))
            }
        } else {
            Err(io::Error::from_raw_os_error(libc::ENOENT))
        }
    }
}

#[polyfuse::async_trait]
impl Filesystem for DebFS {
    async fn call<'a, 'cx, T: ?Sized>(
        &'a self,
        cx: &'a mut Context<'cx, T>,
        op: Operation<'cx>,
    ) -> io::Result<()>
    where
        T: Reader + Writer + Unpin + Send,
    {
        let span = tracing::debug_span!("MemFS::call", unique = cx.unique());
        span.in_scope(|| tracing::debug!(?op));

        macro_rules! try_reply {
            ($e:expr) => {
                match ($e).instrument(span.clone()).await {
                    Ok(reply) => {
                        span.in_scope(|| tracing::debug!(reply=?reply));
                        cx.reply(reply).await
                    }
                    Err(err) => {
                        let errno = err.raw_os_error().unwrap_or(libc::EIO);
                        span.in_scope(|| tracing::debug!(errno=errno));
                        cx.reply_err(errno).await
                    }
                }
            };
        }

        match op {
            Operation::Lookup(op) => try_reply!(self.do_lookup(&op)),
            Operation::Getattr(op) => try_reply!(self.do_getattr(&op)),
            Operation::Read(op) => try_reply!(self.do_read(&op)),
            Operation::Readdir(op) => try_reply!(self.do_readdir(&op)),
            Operation::Readlink(op) => try_reply!(self.do_readlink(&op)),
            _ => {
                span.in_scope(|| tracing::debug!("NOSYS"));
                Ok(())
            }
        }
    }
}

fn dir_entry(kind: u8, name: impl AsRef<OsStr>, ino: u64, off: u64) -> DirEntry {
    let mut e = DirEntry::new(name, ino, off);
    e.set_typ(kind as u32);
    e
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test() -> anyhow::Result<()> {
        let fs = {
            let mut fs = DebFS::new();
            fs.add_deb("testdata/foo_20.04.4_all.deb")?;
            fs
        };

        let inode_id = fs.lookup("etc")?.context("etc not found")?;
        let inode = fs.get(inode_id, || "while looking up etc".into())?;
        assert_eq!(inode.name, OsString::from("etc"));
        assert_eq!(inode.mode, 0o755);
        assert_eq!(inode.kind(), INodeKind::Dir);
        assert_eq!(inode.get_children()?.len(), 1);
        assert_eq!(fs.path(inode_id), OsString::from("etc"));

        let foo_inode_id = inode.get_children()?[0];
        assert_eq!(
            fs.lookup("etc/foo")?.context("etc/foo not found")?,
            foo_inode_id
        );
        let foo_inode = fs.get(foo_inode_id, || "while looking up etc/foo".into())?;
        assert_eq!(foo_inode.name, OsString::from("foo"));
        assert_eq!(foo_inode.mode, 0o644);
        assert_eq!(foo_inode.kind(), INodeKind::File);
        if let INodeContent::File { deb, size } = foo_inode.content {
            assert_eq!(size, 14);
            assert!(fs.debs[deb].ends_with(OsString::from("testdata/foo_20.04.4_all.deb")));
        } else {
            unreachable!();
        }
        assert_eq!(fs.path(foo_inode_id), OsString::from("etc/foo"));
        Ok(())
    }
}
