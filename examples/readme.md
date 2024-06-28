This folder contains some data for running.

## Usage
    |-- demo
        |-- data
            # Data to be processed.
            P.txt
            # Matrix used for converting to ED shape space.
            |-- mesh
                # You can get the surface meshes from our dataset link. 
            |-- points
                # 3D surface points obtained from segmentation. 
                |-- 00.obj
                ...
                |-- 24.obj
        |-- specs.json
        |-- train.json
        |-- test.json
    |-- ini
        |-- C_s
            # Initialized cs parameters. (optional)
            |-- 000_00.pth
            ...
        |-- ini.pth
            # Weights of the pre-trained ED shape model. 
        ...
