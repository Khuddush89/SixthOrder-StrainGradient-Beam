# Updating the existing repository

The ZIP contains the contents for the existing `Khuddush89/SixthOrder-StrainGradient-Beam` repository. Keep its current name and URL.

1. Extract the ZIP and copy the contents of its `SixthOrder-StrainGradient-Beam` folder into your existing local checkout. Replace files with the same names.
2. Remove the old `Greens function.mw` worksheet. It contains the failed saved jump check and is superseded by the verified symbolic script.
3. Run the checks below before committing.
4. Commit and push the updated contents.

```bash
python -m pip install -r requirements.txt
python -m pytest -q code/tests
python code/scripts/verify_green_symbolic.py
python code/scripts/check_package.py
git rm --ignore-unmatch "Greens function.mw"
git add .
git commit -m "Refresh verified beam computations and animated graphics"
git push origin main
```

Copy the folder contents into the repository root. Do not upload the ZIP itself or an extra enclosing folder as the repository source. The package already includes every GIF, PNG and SVG shown in the README.

For GitHub's web upload interface, delete the old worksheet first, then upload the extracted folder contents. The old `Green_Verification.py` is replaced by the corrected compatible entry point.
