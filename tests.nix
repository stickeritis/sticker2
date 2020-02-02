{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  danieldk = pkgs.callPackage sources.danieldk {};
  crateOverrides = with pkgs; defaultCrateOverrides // {
    hdf5-sys = attr: {
      HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };
    };

    sticker2 = attr: {
      BERT_BASE_GERMAN_CASED_VOCAB = sources.bert-base-german-cased-vocab;
    };

    torch-sys = attr: {
      LIBTORCH = "${danieldk.libtorch.v1_4_0}";
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;

    # PyTorch 1.4.0 headers are not compatible with gcc 9. Remove with
    # the next PyTorch release.
    stdenv = if pkgs.stdenv.cc.isGNU then pkgs.gcc8Stdenv else pkgs.stdenv;
  };
  cargo_nix = pkgs.callPackage ./nix/Cargo.nix { inherit buildRustCrate; };
in pkgs.lib.mapAttrsToList (_: drv: drv.build.override {
  features = [ "model-tests" ];
  runTests = true;
}) cargo_nix.workspaceMembers
