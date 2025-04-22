{
  description = "Flask ML API environment with sklearn, pydub, etc.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};

    # Define noisereduce package
    noisereduce = pkgs.python312Packages.buildPythonPackage rec {
      pname = "noisereduce";
      version = "3.0.3"; # Specify the desired version

      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "ff64a28fb92e3c81f153cf29550e5c2db56b2523afa8f56f5e03c177cc5e918f";
      };

      meta = with pkgs.lib; {
        description = "Noise reduction algorithm in Python using spectral gating";
        homepage = "https://github.com/timsainb/noisereduce";
        license = licenses.mit;
      };
    };

    # Define flask_cors package
    flask_cors = pkgs.python312Packages.buildPythonPackage rec {
      pname = "flask_cors";
      version = "5.0.0"; # Specify the desired version

      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "5aadb4b950c4e93745034594d9f3ea6591f734bb3662e16e255ffbf5e89c88ef";
      };

      meta = with pkgs.lib; {
        description = "A Flask extension for handling Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.";
        homepage = "https://github.com/corydolphin/flask-cors";
        license = licenses.mit;
      };

      propagatedBuildInputs = with pkgs.python312Packages; [
        flask
      ];
    };

    pythonPkgs = pkgs.python312.withPackages (ps:
      with ps; [
        flask
        flask_cors
        librosa
        matplotlib
        noisereduce
        numpy
        pydub
        scikit-learn
        scipy
        tqdm
      ]);
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "flask-ml-env";

      packages = [
        pythonPkgs
      ];

      buildInputs = with pkgs; [
        ffmpeg # for pydub audio processing
      ];

      shellHook = ''
        echo "Flask server environment loaded."
        echo "Run in dev mode: flask --app server run"
        echo "Run in dev mode and allow all hosts: flask --app server run --host=0.0.0.0"
      '';
    };
  };
}
