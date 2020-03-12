{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  danieldk = pkgs.callPackage sources.danieldk {};
  # PyTorch 1.4.0 headers are not compatible with gcc 9. Remove with
  # the next PyTorch release.
  stdenv = if pkgs.stdenv.cc.isGNU then pkgs.gcc8Stdenv else pkgs.stdenv;
  crateOverrides = with pkgs; defaultCrateOverrides // {
    hdf5-sys = attr: {
      HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };
    };

    sticker2 = attr: models;

    sentencepiece-sys = attr: {
      nativeBuildInputs = [ pkgconfig ];

      buildInputs = [ (sentencepiece.override (attrs: { inherit stdenv; })) ];
    };

    torch-sys = attr: {
      # Remove when we are not using a git version of tch anymore.
      src = "${attr.src}/torch-sys";

      LIBTORCH = "${danieldk.libtorch.v1_4_0}";
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    inherit stdenv;

    defaultCrateOverrides = crateOverrides;
  };
  cargo_nix = pkgs.callPackage ./nix/Cargo.nix {
    inherit buildRustCrate pkgs stdenv;
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
