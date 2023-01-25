{ pkgs ? import <nixpkgs> {}, ... }:

pkgs.mkShell {

    buildInputs = with pkgs; [

        python310

        python310Packages.ipywidgets
        python310Packages.ipykernel


        python310Packages.pytorchWithoutCuda
        python310Packages.torchvision
        python310Packages.ignite

        python310Packages.numpy
        python310Packages.scipy
        python310Packages.scikitimage
        python310Packages.matplotlib

        

    ];

}