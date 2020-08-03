# We pin nixpkgs to improve reproducability. We don't pin Rust to a
# specific version, but use the latest stable release.

let
  sources = import ./nix/sources.nix;
  models = import ./nix/models.nix;
  nixpkgs = import sources.nixpkgs {};
  sticker = nixpkgs.callPackage sources.sticker {};
  mozilla = nixpkgs.callPackage "${sources.mozilla}/package-set.nix" {};
  libtorch = sticker.libtorch.v1_6_0;
in with nixpkgs; mkShell (models // {
  nativeBuildInputs = [
    mozilla.latest.rustChannels.stable.rust
    pkgconfig
  ];

  buildInputs = [
    curl
    openssl
    sentencepiece
    libtorch
  ] ++ lib.optional stdenv.isDarwin darwin.apple_sdk.frameworks.Security;
  # Unless we use pkg-config, the hdf5-sys build script does not like
  # it if libraries and includes are in different directories.
  HDF5_DIR = symlinkJoin { name = "hdf5-join"; paths = [ hdf5.dev hdf5.out ]; };

  LIBTORCH = "${libtorch.dev}";
})
