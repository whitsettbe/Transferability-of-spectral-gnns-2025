This repository is derived from the ChebNet implementation at [https://github.com/Axeln78/Transferability-of-spectral-gnns](https://github.com/Axeln78/Transferability-of-spectral-gnns). It contains an "Eigenvalue Layer" which directly eigendecomposes graph signals in a precomputed Laplacian eigenbasis (or a similar set of signals) and scales them according to a learned function of the Rayleigh quotients.

Functionality is provided for using the normalized Laplacian ($\tilde{L}=\tilde{U}\tilde{\Lambda}\tilde{U}^T$), standard Laplacian ($L=U\Lambda U^T$), or a post-normalized Laplacian which uses the standard Laplacian eigendecomposition within the normalized expression ($\tilde{L}=D^{-1/2}U\Lambda U^T D^{-1/2}$). In each case, importing external signals to replace the eigenvectors changes the matrix $U$ (or $\tilde{U}$) and its Rayleigh quotients.

Rayleigh quotients can be scaled to the range $[0,2]$ for regularity of the input to the filter functions. If only a partial (low-frequency) spectrum is being used ($numEigs<max_{\mathcal{G}_i=(V_i,E_i)}(|E_i|)$), one can choose to scale the RQs to $[0,2]$ either before (`all`) or after (`sub`) restricting to the subset.

Other hyperparameters used in `Benchmark-gnn/configs/molecults_graph_regression_Eigval_ZINC.json` and the like have roles readily apparent in the codebase. Hyperparameters added since the base ChebNet implementation are guaranteed to match their names between the JSON configs and the `*_Layer` classes (in `Benchmark-gnn/layers`) which use them.

Environment setup can be performed using conda and the included `environment.yml` file.

Externally-sourced harmonics should be stored in `Benchmark-gnn/supp_data`, and should be CSV files formatted according to the output of [https://github.com/czma120/Graph-Signals/blob/main/recursive.py](https://github.com/czma120/Graph-Signals/blob/main/recursive.py).

This codebase has been set up for use with the ZINC 12k dataset, as well as the SBM CLUSTER dataset, although the latter was not tested extensively.

Logs from testing in Summer 2025 are included in `Summer2025Logs`, and various pieces of graph-generation code for side-testing are included in `Summer2025SideTests`