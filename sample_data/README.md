# Sample Data
This is sample data that may be used to test this code. It also serves as a useful reference for how to format data.

## Citations
### Nottingham
Bálint Cserni, Rita Bori, Erika Csörgő, Orsolya Oláh-Németh, Tamás Pancsa, Anita Sejben, István Sejben, András Vörös, Tamás Zombori, Tibor Nyári, Gábor Cserni. ONEST (Observers Needed to Evaluate Subjective Tests) suggests four or more observers for a reliable assessment of the consistency of histological grading of invasive breast carcinoma. Pathol Res Pract. 2022 Jan:229:153718. doi: 10.1016/j.prp.2021.153718. Epub 2021 Dec 6. PMID: 34920295 DOI: 10.1016/j.prp.2021.153718

This data comes out of Figure 1. The following associations are made: `tubulus.csv` with tubule/gland formation (A); `pleomorphism.csv` with nuclear pleomorphism (B); `mitosis.csv` with mitotic activity (C); and `nottingham.csv` with histological grade (D).

### PDL1
Gang Han, Baihong Guo. ONEST: Observers Needed to Evaluate Subjective Tests. https://CRAN.R-project.org/package=ONEST

This data set has gone untested (as of 2024-03-03). I am not sure that the analysis will work properly with data with possible `NA` values or when the datasets in comparison have different dimensions. The original `.RData` may be found in Han and Guo's repository. They note the original data comes from [Reisenbichler et al.][reisenbichler] along with [Rimm et al.][rimm]; I could not verifyingly identify the original sources of the data.

[reisenbichler]: <https://doi.org/10.1038/s41379-020-0544-x> 'Reisenbichler, E. S., Han, G., Bellizzi, A., Bossuyt, V., Brock, J., Cole, K., Fadare, O., Hameed, O., Hanley, K., Harrison, B. T., Kuba, M. G., Ly, A., Miller, D., Podoll, M., Roden, A. C., Singh, K., Sanders, M. A., Wei, S., Wen, H., Pelekanou, V., Yaghoobi, V., Ahmed, F., Pusztai, L., and Rimm, D. L. (2020) “Prospective multi-institutional evaluation of pathologist assessment of PD-L1 assays for patient selection in triple negative breast cancer,” Mod Pathol, DOI: 10.1038/s41379-020-0544-x; PMID: 32300181.'

[rimm]: https://doi.org/10.1001/jamaoncol.2017.0013 'Rimm, D. L., Han, G., Taube, J. M., Yi, E. S., Bridge, J. A., Flieder, D. B., Homer, R., West, W. W., Wu, H., Roden, A. C., Fujimoto, J., Yu, H., Anders, R., Kowalewski, A., Rivard, C., Rehman, J., Batenchuk, C., Burns, V., Hirsch, F. R., and Wistuba,, II (2017) “A Prospective, Multi-institutional, Pathologist-Based Assessment of 4 Immunohistochemistry Assays for PD-L1 Expression in Non-Small Cell Lung Cancer,” JAMA Oncol, 3(8), 1051-1058, DOI: 10.1001/jamaoncol.2017.0013, PMID: 28278348.'

### Prostate
David F Steiner, Kunal Nagpal, Rory Sayres, Davis J Foote, Benjamin D Wedin, Adam Pearce, Carrie J Cai, Samantha R Winter, Matthew Symonds, Liron Yatziv, Andrei Kapishnikov, Trissia Brown, Isabelle Flament-Auvigne, Fraser Tan, Martin C Stumpe, Pan-Pan Jiang, Yun Liu, Po-Hsuan Cameron Chen, Greg S Corrado, Michael Terry, Craig H Mermel. Evaluation of the Use of Combined Artificial Intelligence and Pathologist Assessment to Review and Grade Prostate Biopsies. JAMA Netw Open. 2020 Nov 2;3(11):e2023267. PMID: 33180129 PMCID: PMC7662146 DOI: 10.1001/jamanetworkopen.2020.23267

This data comes out of Figure 2D. The `treatment.csv` data is associated with the AI assisted table while `control.csv` convers the AI unassisted table.