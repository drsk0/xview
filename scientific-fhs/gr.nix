{ stdenv, fetchurl }:

stdenv.mkDerivation {
  name = "gr";
  src = fetchurl {
    url =
      "https://github.com/sciapp/gr/releases/download/v0.69.1/gr-0.69.1-Linux-x86_64.tar.gz";
    sha256 = "sha256-nNq4djmqvwQDF49WzdMvxHnU6CDdOJ9PUbNZfqbWLG8=";
  };
  installPhase = ''
    mkdir $out
    cp -R * $out/
  '';
  dontStrip = true;
}
