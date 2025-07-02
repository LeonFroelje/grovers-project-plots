{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }: 
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
      devShells.${system} = {
        default = (pkgs.buildFHSEnv {
          name = "qiskit";
          targetPkgs = pkgs: (with pkgs; [
            python312
            python312Packages.pip
            zsh
            zlib
          ]);
          runScript = ''
          zsh
          '';
        }).env;
    };
  };
}
