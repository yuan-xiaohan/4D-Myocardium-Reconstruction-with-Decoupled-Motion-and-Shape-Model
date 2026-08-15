# Third-party notices

The minimal mask canonicalization in `preprocess_mask.py` is adapted from the
standalone inference utilities developed for UVRecons. Only NIfTI validation,
contour extraction, rigid registration, and orientation scoring are retained.
The standalone reconstruction and MRI-segmentation models are not included.

The compact `assets/registration_template.npz` is derived from the project's
canonical cardiac registration templates. It contains only point arrays needed
to establish the coordinate frame used by the released reconstruction models.

Runtime dependencies retain their respective upstream licenses.
