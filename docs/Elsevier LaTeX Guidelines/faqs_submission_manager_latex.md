# FAQs: Submission Manager LaTeX

## How to identify and fix errors from LaTeX error codes in the built PDF?

Please check the publication FAQ for more detailed information.

## Do I need to include a style (.sty) file?

Yes. Editorial Manager (EM) requires a `.sty` file to properly compile your submission. Check the journal homepage, because the publication may provide a standard style file that you should include when submitting your manuscript.

## EM cannot compile my LaTeX submission containing EPS image files. What is wrong?

EPS files created by authors can be compiled in many different ways.

Different programs use different standards for creating EPS files. Open EPS files in Adobe Illustrator or a similar image editing program when possible. Opening EPS files in Adobe Photoshop and saving them in a different file format may help, but saving vector files in Photoshop can compromise image quality. Try submitting your image files again after saving them to a new format such as `.jpg` or `.png`.

## Can I use embedded fonts in an EPS image file?

No. EM cannot process embedded fonts in EPS image files. Remove embedded fonts before uploading EPS files. Alternatively, convert the EPS file locally to PDF and upload the PDF file to EM.

## What should I do if my LaTeX files cannot be compiled?

Click `View Submission` to open the PDF and check whether an error message appears in the TeX compiler log file shown in the PDF. If your submission files did not compile properly, you will see one or more pages containing TeX Live error codes. The header will be similar to this:

```text
pdfTeX, Version 3.141592653-2.6-1.40.24 (TeX Live 2022/W32TeX)
```

The TeX compiler log information will appear below this line. This indicates the point at which an error occurred that prevented the file from being compiled. A typical error looks like this:

```text
! LaTeX Error: File 'aries.sty' not found.

Type X to quit or <RETURN> to proceed, or enter new name.

(Default extension: sty)

Enter file name:

! Emergency stop.
```

In this example, the `aries.sty` style file is missing. The submission cannot be compiled without the style file referenced in the primary `.tex` manuscript file.

Follow these steps to add the missing style file:

1. Click `Edit Submission` and upload the missing style file as a `Manuscript` item.
2. Do not select the `Supplemental` item type for the style file.
3. Rebuild the submission PDF.
4. View the new PDF and verify that the LaTeX submission compiled correctly and is formatted properly.

If the submission still cannot be compiled after adding the missing style file, check the following:

- Make sure images are referenced correctly in the `.tex` manuscript file.
- Make sure images are not stored in subfolders. EM cannot process LaTeX submissions that contain subfolders.
- Make sure all other accompanying submission files are referenced correctly in the `.tex` file.

## Which file types are excluded by EM?

The following filenames cannot be uploaded to EM because they are reserved names within the EM environment:

- `CON`
- `PRN`
- `AUX`
- `NUL`
- `COM1`
- `COM2`
- `COM3`
- `COM4`
- `COM5`
- `COM6`
- `COM7`
- `COM8`
- `COM9`
- `LPT1`
- `LPT2`
- `LPT3`
- `LPT4`
- `LPT5`
- `LPT6`
- `LPT7`
- `LPT8`
- `LPT9`

Avoid using these names even with extensions, for example `NUL.txt`.

Files with special characters in the filename are also excluded, for example `G+.eps`.

Filenames should contain only one period to mark the extension. For example:

- Incorrect: `fig.1.eps`
- Correct: `fig1.eps`

## Can I upload a file in a compressed ZIP or RAR format?

EM supports `.zip` files and `.tar.gz` archives containing the complete LaTeX submission set. EM does not support the RAR archive format.

## Why does my PDF show question marks instead of bibliographic citations?

If you compiled your bibliography in a separate file and question marks appear in the PDF instead of the bibliography content, your `.tex` manuscript file may contain incorrect formatting.

Try one of the following methods.

### Method A: bibliography is a `.bib` file

Example: `reference.bib`

```tex
\bibliographystyle{spmpsci}
\bibliography{reference}
```

If the bibliography still does not appear when using the `.bib` file, try uploading the author's `.bbl` file and use Method B.

### Method B: bibliography is a `.bbl` file

Example: `Blockseminar_opt_trans.bbl`

```tex
\input{Blockseminar_opt_trans.bbl}
```

## Which TeX implementations and packages are supported by EM?

The PDF builders determine which TeX engine to run based on the content of your LaTeX submission. EM includes support for:

- `LaTeX`
- `pdfLaTeX`
- `pdfTeX`
- `TeX`
- `XeLaTeX`

EM first attempts to build the PDF using `LaTeX`. If that build does not succeed, the system tries `pdfLaTeX`. Compilation failures are indicated in the EM submission PDF log.

Note: Disable all non-standard plug-ins in your local TeX installation when creating your submission.

## Can I use the same filenames for both PDF and LaTeX submission files?

No. EM cannot process a submission containing two or more files with the same base name, even if their extensions differ. This can cause PDF generation errors.

## Which item types should I select for LaTeX submission files?

Select the `Manuscript` item type for these file types:

- `.tex`
- `.bbl`
- `.bst`
- `.sty`
- `.bib`
- `.nls`
- `.ilg`
- `.nlo`

Select the `Figure` item type for images and graphics files.

Depending on the publication, `Manuscript`, `Figure`, or other item types may have different names. Contact the support team through the journal's contact options if you need clarification.

Do not upload LaTeX files as `Supplemental` items.

## Can I use Overleaf to create my LaTeX manuscript and import it into EM?

Yes. You can use Overleaf to create your LaTeX manuscript using TeX Live 2022. When using Overleaf, inspect the generated PDF carefully to ensure the source compiled without errors.

Unlike EM, Overleaf may still generate a PDF even when compilation errors occurred. EM, by contrast, can produce only an error-log PDF when compilation fails.

Elsevier Overleaf templates:

- <https://www.overleaf.com/latex/templates?q=elsevier+template>

Overleaf tutorial:

- <https://www.overleaf.com/learn/latex/Tutorials>

## Can I use subfolders in my TeX submission files?

No. EM cannot process LaTeX submissions containing subfolders. All submission files, including figures, tables, style files, and bibliography files, must be stored at the same folder level.

Example layout:

```text
my-latex-submission/
  LaTeX_sample_file.tex
  Other_LaTeX-files.tex
  Bibliography.bib
  Bibliography_file.bbl
  nomenclature.nls
  nomenclature.ilg
  nomenclature.nlo
  Sample.sty
  Sample.cls
  Sample.bbl
  Sample.bst
  image_files.eps
  image_files.png
  image_files.pdf
  additional_submission_file.tex
```

Files such as `.sty`, `.cls`, `.bbl`, or `.bst` are only needed when they are not already available in the EM TeX installation.

## Which image formats are supported for LaTeX submissions?

EM supports many image formats, including:

- `.png`
- `.pdf`
- `.jpg`
- `.mps`
- `.jpeg`
- `.jbig2`
- `.jb2`
- `.eps`
- `.ps`
- `.tif`

Note: `.tif` files cannot be used inside the `\includegraphics` command.

## Why do I see question marks instead of references to tables and figures?

If question marks appear in the PDF instead of referenced tables or figures, you may have submitted a compressed file that contains one or more subdirectories. EM cannot process compressed LaTeX submissions that include subfolders. All related submission files inside a compressed archive must be stored at the same directory level.

## Why are my figures not appearing in the PDF?

A common cause is that images were stored in subfolders. EM cannot process LaTeX submissions with directory structures, so it cannot resolve files referenced outside the root folder.

Correct example:

```tex
\epsfig{figure=alld.eps,width=.5\textwidth}
```

Incorrect example:

```tex
\epsfig{figure=images/alld.eps,width=.5\textwidth}
```

## Why can EM not compile my XeLaTeX submission?

EM makes an automatic determination of the TeX package or engine to use. If the PDF builder does not compile your files correctly, click `View Submission` and inspect the compiler log shown in the PDF.

You may see a message similar to this:

```text
Package: fontenc 2022-06-14 v2.1 Standard LaTeX package
and/or
! LaTeX Error: Cannot determine size of graphic in fig5.pdf (no BoundingBox)
```

To help EM build the PDF correctly, insert this directive on line 1 of your `.tex` manuscript file:

```tex
%!TEX TS-program = xelatex
```

Then save the `.tex` file under a different filename from the original and replace the uploaded `.tex` file with the new one. Rebuild the submission PDF. EM should then be able to process the XeLaTeX submission successfully.
