# We pin nixpkgs to improve reproducability. We don't pin Rust to a
# specific version, but use the latest stable release.

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  nixpkgs = import sources.nixpkgs {};
  danieldk = nixpkgs.callPackage sources.danieldk {};
  mozilla = nixpkgs.callPackage "${sources.mozilla}/package-set.nix" {};

  # PyTorch 1.4.0 headers are not compatible with gcc 9. Remove with
  # the next PyTorch release.
  stdenv = if nixpkgs.stdenv.cc.isGNU then nixpkgs.gcc8Stdenv else nixpkgs.stdenv;
  mkShell = nixpkgs.mkShell.override (attr: { inherit stdenv; });
in with nixpkgs; mkShell (models // {
  nativeBuildInputs = [
    mozilla.latest.rustChannels.stable.rust
    pkgconfig
    protobuf
  ];

  buildInputs = [
    curl
    openssl
    (sentencepiece.override (attrs: { inherit stdenv; }))
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;
  # Unless we use pkg-config, the hdf5-sys build script does not like
  # it if libraries and includes are in different directories.
  HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };

  LIBTORCH = "${danieldk.libtorch.v1_4_0}";
})
