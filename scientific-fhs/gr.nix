{ stdenv, fetchurl, qt5, libGL }:

stdenv.mkDerivation {
  name = "gr";
  src = fetchurl {
    url =
      "https://github.com/sciapp/gr/releases/download/v0.70.0/gr-0.70.0-Linux-x86_64.tar.gz";
    sha256 = "sha256-/SL/iJFOkzbxXFu5uPsk6HbiAgiZvQe3pBfyuC5m8UQ=";
  };
  installPhase = ''
    mkdir $out
    cp -R * $out/
    cp ${qt5.qtbase.out}/lib/libQt5Widgets.so.5 $out/lib/
    cp ${qt5.qtbase.out}/lib/libQt5Gui.so.5 $out/lib/
    cp ${qt5.qtbase.out}/lib/libQt5Network.so.5 $out/lib/
    cp ${qt5.qtbase.out}/lib/libQt5Core.so.5 $out/lib/
    cp ${libGL.out}/lib/libGL.so.1 $out/lib/
    cp ${stdenv.cc.cc.lib}/lib/libstdc++.so.6 $out/lib/;
  '';
  dontStrip = true;
}
