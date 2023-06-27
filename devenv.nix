{ pkgs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = [ pkgs.git ];

  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo hello from $GREET";

  enterShell = ''
    hello
    git --version
  '';

  # https://devenv.sh/languages/
  languages.nix.enable = true;
  languages.julia.enable = true;

  scripts.patch-gksqt.exec = 
      let
        libPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib pkgs.qt5.qtbase pkgs.libGL ];
      in
      ''
        GRDIR=$(julia --project=. -e 'import Pkg; Pkg.add("Plots"); using Plots; println(ENV["GRDIR"])' | awk -F'/src' '{print $1}')
        echo $GRDIR
        export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins"
        ${pkgs.patchelf}/bin/patchelf --set-interpreter \${pkgs.glibc}/lib/ld-linux-x86-64.so.2 --set-rpath "\${libPath}" $GRDIR/bin/gksqt
        ldd $GRDIR/bin/gksqt | grep -q "not found" || echo '../gksqt is patched'
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
