{
  pkgs ? import (import ./nix/sources.nix).nixpkgs {
    config = {
      allowUnfreePredicate = pkg: builtins.elem (pkgs.lib.getName pkg) [
        "libtorch"
      ];
    };
  }
}:

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  crateOverrides = with pkgs; defaultCrateOverrides // {
    hdf5-sys = attr: {
      HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };
    };

    sticker2 = attr: {
      buildInputs = [ libtorch-bin ] ++
        lib.optional stdenv.isDarwin darwin.Security;
    } // models;

    sticker2-utils = attr: {
      buildInputs = [ libtorch-bin ] ++
        lib.optional stdenv.isDarwin darwin.Security;
    };

    sentencepiece-sys = attr: {
      nativeBuildInputs = [ pkgconfig ];

      buildInputs = [ sentencepiece ];
    };

    torch-sys = attr: {
      buildInputs = lib.optional stdenv.isDarwin curl;

      LIBTORCH = "${libtorch-bin.dev}";
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
  crateTools = import "${sources.crate2nix}/tools.nix" {};
  cargoNix = pkgs.callPackage (crateTools.generatedCargoNix {
    name = "sticker2";
    src = pkgs.nix-gitignore.gitignoreSource [ ".git/" "nix/" "*.nix" ] ./.;
  }) {
    inherit buildRustCrate;
  };
in with pkgs; lib.flatten (lib.mapAttrsToList (_: drv: [
  (drv.build.override {
    features = [ "model-tests" ];
    runTests = true;
  })
  (drv.build.override {
    features = [ "load-hdf5" "model-tests" "tensorboard" ];
    runTests = true;
  })
]) cargoNix.workspaceMembers)
