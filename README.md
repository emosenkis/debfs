# DebFS - virtually unpack Debian packages

## Status

- Currently provides a read-only filesystem with the specified .deb files 'unpacked' into it.
- Metadata is stored in `<mount-point>/.debfs` (in the parent filesystem)
  - Once a .deb is added, it does not need to be added when remounting at the same mount point.
  - Currently, the only way to remove a deb is to unmount the FS and delete `<mount-point>/.debfs`

## Missing features

- Better unit tests.
- Support chrooting into a DebFS (currently fails with a permissions error).
- Permit modifying the FS (by passing through to an underlying filesystem).
- Dependency resolution.
- Config management.
- Install/remove packages from a mounted DebFS.
- Smart uninstall (returns the FS to a pristine state as if the package had never been installed).
- Support file types beyond the currently supported regular files, directories, and symlinks.
- Verify checksums

## Usage

``` shell
debfs -m <mount-point> [foo.deb] [bar.deb] ...
```

## External Dependencies

- liblzma
  
  The following works on Ubuntu 20.04:
  
  ``` shell
  sudo apt-get install liblzma-dev
  ```
