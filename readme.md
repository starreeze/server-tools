# xpkg

xpkg - a minimal package management tool without root

version 0.3 by *xsy*, GPLv3 License

## requirements

ubuntu >= 16.04

python >= 3.5

## features

- No root needed;

- automatically solve dependencies;

- simple and convenient.

## usage

### install

Before the first use, run

```
SOFTWARE_BASE=<path> python xpkg.py
```

in which `<path>` is an arbitrary writable location where you want to put all the packages to be installed.

After complete, you'll see a notice asking you to put something into you .bashrc file (or whatever shell config file). Just do it manually.

Or if already installed an older version, you can run

```
SOFTWARE_BASE=<path> python xpkg.py --update
```

after obtaining the latest version.

### command

After installation, you can use xpkg at your will.

```
xpkg --help           # show help
xpkg --version        # show version
xpkg --list           # list all installed packages
xpkg -i unar aria2    # install packages
xpkg -i unar --force  # force install, i.e. ignoring dependency problems
xpkg -r unar aria2    # remove packages
xpkg --clear          # uninstall xpkg and clear all packages
```

Enjoy!

## Known issues

This script depends on the apt source lists to get packages. However, sometimes these lists are outdated and therefore 404 errors would occur.

The best solution is to ask your administrator to run `apt update`. But if you can, why not just ask him to install the packages for you?

In this case you can try with `--force`, and if that fails we really have nothing to do about it.
