{
  description = "Julia development environment";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; allowBroken = true; permittedInsecurePackages =
        ["libgit2-0.27.10"]; };
        # overlays = [ self.overlay ];
      };


      fhsCommand = pkgs.callPackage ./scientific-fhs {
          juliaVersion = "julia_16";
      };
    in
    {

      devShell.${system} = pkgs.mkShell rec {
        buildInputs = [
          pkgs.aria
          pkgs.qgis
          (fhsCommand "julia" "julia")
          (fhsCommand "julia-bash" "bash")
        ];

        # CPATH = pkgs.lib.makeSearchPathOutput "dev" "include" buildInputs;
        # shellHook = ''
        # '';
      };
    };
}
