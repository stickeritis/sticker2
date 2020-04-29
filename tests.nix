{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  danieldk = pkgs.callPackage sources.danieldk {};
  libtorch = danieldk.libtorch.v1_5_0;
  crateOverrides = with pkgs; defaultCrateOverrides // {
    hdf5-sys = attr: {
      HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };
    };

    sticker2 = attr: {
      buildInputs = [ libtorch ] ++
        lib.optional stdenv.isDarwin darwin.Security;
    } // models;

    sticker2-utils = attr: {
      buildInputs = [ libtorch ] ++
        lib.optional stdenv.isDarwin darwin.Security;
    };

    sentencepiece-sys = attr: {
      nativeBuildInputs = [ pkgconfig ];

      buildInputs = [ (sentencepiece.override (attrs: { inherit stdenv; })) ];
    };

    torch-sys = attr: {
      buildInputs = lib.optional stdenv.isDarwin curl;

      LIBTORCH = "${libtorch.dev}";
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
  cargo_nix = pkgs.callPackage ./nix/Cargo.nix {
    inherit buildRustCrate;
  };
in with pkgs; lib.flatten (lib.mapAttrsToList (_: drv: [
  (drv.build.override {
    features = [ "model-tests" ];
    runTests = true;
  })
  (drv.build.override {
    features = [ "load-hdf5" "model-tests" ];
    runTests = true;
  })
]) cargo_nix.workspaceMembers)
