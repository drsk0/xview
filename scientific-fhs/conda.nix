{ runCommand, fetchurl, makeWrapper, installationPath }:

let
  conda_version = "4.7.12.1";
  conda_src = fetchurl {
    url =
      "https://repo.continuum.io/miniconda/Miniconda3-${conda_version}-Linux-x86_64.sh";
    sha256 = "sha256-v+NOH6KNbXWnrQX9AvpUcidWc9X1Yht3OAiY3uG+FdI=";
  };

in runCommand "conda-install" { buildInputs = [ makeWrapper ]; } ''
  mkdir -p $out/bin
  cp ${conda_src} $out/bin/miniconda-installer.sh
  chmod +x $out/bin/miniconda-installer.sh
  makeWrapper                            \
    $out/bin/miniconda-installer.sh      \
    $out/bin/conda-install               \
    --add-flags "-p ${installationPath}" \
    --add-flags "-b"
''
