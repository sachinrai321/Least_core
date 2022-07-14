Search.setIndex({"docnames": ["getting-started", "index", "install", "notebooks/index", "notebooks/knn_shapley", "valuation/cli", "valuation/index", "valuation/influence", "valuation/influence/general", "valuation/influence/linear", "valuation/influence/types", "valuation/loo", "valuation/loo/naive", "valuation/models", "valuation/models/binary_logistic_regression", "valuation/models/linear_regression_torch_model", "valuation/models/pytorch_model", "valuation/reporting", "valuation/reporting/plots", "valuation/reporting/scores", "valuation/shapley", "valuation/shapley/knn", "valuation/shapley/montecarlo", "valuation/shapley/naive", "valuation/solve", "valuation/solve/cg", "valuation/utils", "valuation/utils/caching", "valuation/utils/dataset", "valuation/utils/logging", "valuation/utils/numeric", "valuation/utils/parallel", "valuation/utils/plotting", "valuation/utils/progress", "valuation/utils/types", "valuation/utils/utility"], "filenames": ["getting-started.rst", "index.rst", "install.rst", "notebooks/index.rst", "notebooks/knn_shapley.ipynb", "valuation/cli.rst", "valuation/index.rst", "valuation/influence.rst", "valuation/influence/general.rst", "valuation/influence/linear.rst", "valuation/influence/types.rst", "valuation/loo.rst", "valuation/loo/naive.rst", "valuation/models.rst", "valuation/models/binary_logistic_regression.rst", "valuation/models/linear_regression_torch_model.rst", "valuation/models/pytorch_model.rst", "valuation/reporting.rst", "valuation/reporting/plots.rst", "valuation/reporting/scores.rst", "valuation/shapley.rst", "valuation/shapley/knn.rst", "valuation/shapley/montecarlo.rst", "valuation/shapley/naive.rst", "valuation/solve.rst", "valuation/solve/cg.rst", "valuation/utils.rst", "valuation/utils/caching.rst", "valuation/utils/dataset.rst", "valuation/utils/logging.rst", "valuation/utils/numeric.rst", "valuation/utils/parallel.rst", "valuation/utils/plotting.rst", "valuation/utils/progress.rst", "valuation/utils/types.rst", "valuation/utils/utility.rst"], "titles": ["Getting started", "pyDVL", "Installing pyDVL", "Notebooks", "KNN Shapley", "cli", "valuation", "influence", "general", "linear", "types", "loo", "naive", "models", "binary_logistic_regression", "linear_regression_torch_model", "pytorch_model", "reporting", "plots", "scores", "shapley", "knn", "montecarlo", "naive", "solve", "cg", "utils", "caching", "dataset", "logging", "numeric", "parallel", "plotting", "progress", "types", "utility"], "terms": {"In": [0, 2], "order": [0, 2, 20, 22, 26, 27, 30], "us": [0, 2, 8, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 33], "librari": [0, 1, 26, 28, 31], "you": [0, 1, 2, 30], "need": [0, 4, 14, 15, 26, 27, 31], "memcach": [0, 26, 27, 30], "an": [0, 10, 16, 18, 20, 22, 25, 26, 28, 30, 31, 34], "memori": [0, 31], "kei": [0, 18, 19, 26, 27], "store": 0, "small": [0, 26, 27], "chunk": [0, 26, 31], "arbitrari": 0, "data": [0, 1, 4, 8, 9, 12, 18, 19, 20, 21, 22, 26, 27, 28, 31, 35], "string": 0, "object": [0, 16, 20, 21, 22, 26, 27, 28, 31, 33, 34, 35], "cach": [0, 1, 6, 26, 31], "certain": [0, 28], "result": [0, 18, 25, 26, 27, 31], "speed": 0, "up": [0, 8, 9, 10, 26, 30], "can": [0, 1, 2, 8, 30, 31], "either": [0, 8, 26, 33], "instal": [0, 1], "directli": [0, 4], "your": 0, "system": 0, "run": [0, 4, 14, 15, 18, 20, 22, 26, 27, 30, 31], "For": [0, 18, 20, 22], "refer": [0, 26, 28], "section": 0, "s": [0, 1, 18, 26, 27, 30], "wiki": [0, 25], "u": [0, 20, 22, 23, 26, 35], "user": 0, "Or": [0, 26, 31], "insid": 0, "contain": [0, 8, 9], "docker": 0, "rm": 0, "p": 0, "11211": [0, 26, 27], "latest": [0, 2], "v": [0, 4, 16, 25, 30, 34], "enabl": 0, "default": [0, 26, 27, 31], "disabl": [0, 18, 26, 27], "desir": 0, "import": [0, 2, 4, 26, 27, 30], "numpi": [0, 2, 4], "np": [0, 4, 8, 9, 25, 26, 30, 31], "from": [0, 4, 18, 19, 20, 21, 22, 25, 26, 28, 29, 30], "valuat": [0, 1, 2, 4, 26, 27, 28, 29, 30, 31], "sklearn": [0, 4, 19, 26, 28], "model_select": 0, "train_test_split": 0, "x": [0, 14, 15, 16, 18, 26, 28, 30, 31, 34], "y": [0, 9, 16, 26, 30, 34], "arang": 0, "100": [0, 20, 22], "reshap": 0, "50": 0, "2": [0, 4, 10, 16, 25, 26, 30, 31], "x_train": [0, 8, 9, 26, 28], "x_test": [0, 8, 9, 26, 28], "y_train": [0, 8, 9, 26, 28], "y_test": [0, 8, 9, 26, 28], "test_siz": 0, "0": [0, 4, 18, 19, 20, 22, 25, 26, 27, 28, 30, 31, 35], "5": [0, 4, 28], "random_st": [0, 26, 28], "16": 0, "linear_model": [0, 9, 30], "linearregress": 0, "model": [0, 1, 6, 8, 9, 12, 16, 19, 20, 21, 22, 26, 27, 35], "map_reduc": [0, 26, 31], "montecarlo": [0, 1, 6, 20, 30], "combinatorial_montecarlo_shaplei": [0, 20, 22], "report": [0, 1, 6, 26, 28, 31], "score": [0, 1, 6, 12, 16, 17, 20, 22, 26, 27, 34, 35], "compute_fb_scor": [0, 19], "plot": [0, 1, 4, 6, 17, 26], "shapley_result": [0, 18], "fun": [0, 26, 31], "partial": 0, "truncated_montecarlo_shaplei": [0, 20, 22], "progress": [0, 1, 6, 8, 12, 16, 19, 20, 21, 22, 23, 26, 31, 34, 35], "true": [0, 4, 12, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 33, 35], "values_nmc": 0, "hist_nmc": 0, "num_run": [0, 18, 26, 31], "10": [0, 26, 31], "num_job": [0, 20, 22, 26, 30, 31], "160": 0, "scores_nmc": 0, "welcom": 1, "find": 1, "repositori": 1, "here": [1, 18, 26, 34, 35], "get": [1, 8, 30], "start": [1, 18, 31], "creat": [1, 26, 31], "dataset": [1, 4, 6, 12, 19, 20, 21, 22, 26, 32, 35], "util": [1, 4, 6, 20, 22, 23, 27, 29, 30, 31], "comput": [1, 12, 14, 15, 18, 19, 20, 21, 22, 23, 26, 27, 30], "shaplei": [1, 3, 6, 19, 21, 22, 23, 26, 28, 30], "valu": [1, 4, 10, 16, 18, 19, 20, 21, 22, 23, 26, 27, 28, 30, 31, 33], "depend": 1, "notebook": [1, 4], "knn": [1, 3, 6, 20], "cli": [1, 6], "influenc": [1, 2, 6, 8, 9], "gener": [1, 6, 7, 26, 27, 30, 31], "linear": [1, 6, 7, 25, 30], "type": [1, 6, 7, 15, 26, 31], "loo": [1, 6, 12], "naiv": [1, 6, 11, 20, 22, 33], "binary_logistic_regress": [1, 6, 13], "linear_regression_torch_model": [1, 6, 13], "pytorch_model": [1, 6, 13], "solv": [1, 6], "cg": [1, 6, 8, 24], "log": [1, 6, 26], "numer": [1, 6, 26], "parallel": [1, 6, 19, 22, 26, 30], "index": [1, 18, 20, 22, 26, 28, 31], "search": 1, "page": 1, "requir": [2, 30], "follow": [2, 29], "python": [2, 29], "3": [2, 4, 26, 27, 29], "8": [2, 4, 25, 26, 28, 29], "scikit": 2, "learn": 2, "joblib": [2, 31], "pymemcach": [2, 26, 27], "tqdm": [2, 20, 22, 26, 31, 33], "matplotlib": [2, 4, 18], "option": [2, 8, 9, 14, 15, 16, 18, 25, 26, 27, 28, 30, 31, 32, 35], "want": 2, "function": [2, 14, 15, 20, 22, 25, 26, 27, 28, 30, 31], "also": [2, 25], "pytorch": 2, "To": [2, 31], "releas": 2, "pip": 2, "If": [2, 26, 27, 31], "should": [2, 14, 15, 25, 28], "check": [2, 25, 30], "c": 2, "print": 2, "__version__": 2, "1": [4, 8, 10, 16, 18, 20, 22, 25, 26, 27, 28, 30, 31, 34], "load_ext": 4, "autoreload": 4, "os": 4, "sy": 4, "pathlib": 4, "path": 4, "pyplot": 4, "plt": [4, 18], "neighbor": 4, "kneighborsclassifi": [4, 21], "exact_knn_shaplei": [4, 21], "add": [4, 19, 20, 22, 26, 31], "directori": 4, "abl": 4, "py": [4, 21, 25, 26, 28, 30], "first": [4, 20, 22], "one": [4, 8, 14, 15, 20, 22, 26, 28, 31], "when": [4, 20, 22, 26, 27, 28], "insert": 4, "fspath": 4, "resolv": 4, "second": [4, 26, 27], "test": [4, 8, 9, 19, 20, 22, 29], "4": [4, 29], "plot_iri": 4, "from_sklearn": [4, 26, 28], "load_iri": 4, "6": 4, "n_neighbor": 4, "7": 4, "effect": 4, "k": [4, 25, 30], "item": [4, 19, 27], "ab": 4, "arrai": 4, "9": [4, 26, 31], "plot_test": 4, "maybe_init_task": 5, "task_nam": 5, "str": [5, 8, 18, 26, 27, 28, 29, 31, 32, 34], "clearml_config": 5, "dict": [5, 16, 18, 19, 20, 22, 26, 27, 32, 34], "task_param": 5, "sourc": [5, 8, 9, 10, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "fixm": [5, 26, 27, 31], "task": [5, 31], "connect": 5, "work": [5, 26, 28], "copi": [5, 22, 28, 31], "param": [5, 25, 26, 31], "twicedifferenti": [8, 34], "ndarrai": [8, 9, 14, 15, 16, 18, 19, 25, 26, 28, 30, 32, 34], "none": [8, 9, 14, 15, 16, 18, 20, 22, 25, 26, 27, 28, 29, 30, 31, 32, 35], "bool": [8, 12, 14, 15, 16, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 33, 34, 35], "fals": [8, 16, 20, 22, 25, 26, 27, 28, 30, 34, 35], "n_job": 8, "int": [8, 12, 14, 15, 16, 18, 19, 20, 22, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35], "influence_typ": [8, 9], "influencetyp": [8, 9, 10], "inversion_method": 8, "max_data_point": 8, "calcul": [8, 9, 25, 26, 27, 28, 34], "train": [8, 9, 12, 14, 15, 19, 26, 27, 28], "point": [8, 9, 12, 19, 20, 22, 27, 28], "j": 8, "i": [8, 18, 20, 22, 30, 31, 34], "matrix": [8, 18, 25], "i_": 8, "ij": 8, "It": [8, 25, 26, 30, 31], "doe": [8, 26, 31], "so": [8, 20, 22, 27], "factor": 8, "all": [8, 14, 15, 20, 22, 26, 27, 30, 31, 34], "respect": [8, 9, 25, 30, 34], "subsequ": 8, "over": [8, 20, 22, 30, 34], "set": [8, 9, 18, 19, 20, 22, 26, 27, 28, 30, 31, 33], "paramet": [8, 9, 18, 19, 20, 21, 22, 26, 27, 28, 30, 31, 33, 34], "A": [8, 9, 25, 26, 27, 28, 30, 31, 33, 34, 35], "which": [8, 25, 26, 31, 33], "ha": [8, 25, 26, 27], "implement": [8, 19, 20, 22, 25, 31], "interfac": [8, 22], "shape": [8, 9, 25, 30], "mxk": [8, 9], "featur": [8, 9, 26, 28], "mxl": [8, 9], "target": [8, 9, 25, 26, 28], "nxk": [8, 9], "nxl": [8, 9], "whether": [8, 20, 21, 22, 30], "displai": [8, 20, 21, 22, 26, 31, 33], "bar": [8, 19, 20, 21, 22, 26, 31, 33], "The": [8, 18, 21, 25, 26, 27, 31], "number": [8, 18, 20, 22, 25, 26, 27, 28, 30, 31], "job": [8, 20, 22, 26, 31], "process": [8, 20, 22, 26, 29, 31], "perturb": [8, 10], "invers": 8, "method": [8, 20, 22, 25, 26, 27, 31, 33], "specif": 8, "direct": 8, "explicit": 8, "construct": [8, 26, 28], "hessian": [8, 30, 34], "conjug": [8, 25], "gradient": [8, 25, 34], "return": [8, 9, 19, 20, 22, 25, 26, 27, 28, 30, 31, 33], "nxm": [8, 9, 30], "specifi": 8, "linear_influ": 9, "onto": 9, "valid": 9, "assum": [9, 30], "ax": [9, 25], "b": [9, 25, 26, 30, 34], "bxc": 9, "influences_up_linear_regression_analyt": 9, "tupl": [9, 14, 15, 20, 22, 26, 28, 30, 32, 35], "n": [9, 25, 30, 31], "repres": [9, 25, 30], "influences_perturbation_linear_regression_analyt": 9, "each": [9, 12, 18, 19, 20, 22, 26, 28, 29, 30, 31], "bxcxm": 9, "class": [10, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35], "base": [10, 14, 15, 16, 26, 27, 28, 29, 30, 31, 33, 34, 35], "enum": [10, 16, 30], "enumer": [10, 16, 26, 30, 33], "naive_loo": 12, "supervisedmodel": [12, 19, 26, 34, 35], "ordereddict": [12, 18, 19, 20, 21, 22, 23], "float": [12, 16, 18, 19, 20, 22, 25, 26, 27, 28, 30, 31, 34, 35], "binarylogisticregressiontorchmodel": 14, "n_input": 14, "init": [14, 15], "modul": [14, 15, 16, 26, 28, 31, 34], "forward": [14, 15, 18, 19, 26, 33], "tensor": [14, 15], "defin": [14, 15], "perform": [14, 15, 20, 22], "everi": [14, 15, 20, 22], "call": [14, 15, 18, 21, 26, 28, 31], "overridden": [14, 15, 31], "subclass": [14, 15, 31], "although": [14, 15], "recip": [14, 15], "pass": [14, 15, 26, 31], "within": [14, 15, 20, 22], "thi": [14, 15, 20, 22, 26, 27, 28, 29, 30, 31], "instanc": [14, 15], "afterward": [14, 15], "instead": [14, 15, 20, 22, 25, 26, 27], "sinc": [14, 15], "former": [14, 15], "take": [14, 15, 26, 27, 28, 30, 31], "care": [14, 15], "regist": [14, 15], "hook": [14, 15], "while": [14, 15], "latter": [14, 15], "silent": [14, 15], "ignor": [14, 15, 26, 33], "them": [14, 15, 26, 31], "lrtorchmodel": 15, "dim": [15, 26, 28], "tt": 16, "flatten_gradi": 16, "grad": [16, 34], "pytorchoptim": 16, "adam": 16, "adam_w": 16, "pytorchsupervisedmodel": 16, "torchobject": [16, 34], "optim": 16, "optimizer_kwarg": 16, "num_epoch": 16, "batch_siz": 16, "64": 16, "num_param": [16, 34], "mvp": [16, 34], "second_x": [16, 34], "kwarg": [16, 18, 26, 31, 34], "fit": [16, 19, 26, 34], "predict": [16, 26, 34], "shaded_mean_std": 18, "color": [18, 32], "num_std": 18, "usual": 18, "mean": [18, 20, 22, 26, 27, 31], "std": [18, 26, 27], "deviat": 18, "aggreg": [18, 26, 31], "axi": 18, "e": [18, 20, 22, 26, 30, 31, 34], "standard": 18, "shade": 18, "around": 18, "ar": [18, 20, 22, 26, 27, 30, 31], "filenam": 18, "savefig": 18, "save": 18, "exampl": 18, "dictionari": [18, 31], "all_valu": 18, "num_point": 18, "backward_scor": 18, "backward_scores_revers": 18, "backward_random_scor": 18, "forward_scor": 18, "forward_scores_revers": 18, "forward_random_scor": 18, "max_iter": [18, 20, 22, 25], "score_nam": 18, "spearman_correl": 18, "vv": 18, "list": [18, 19, 20, 22, 26, 28, 30, 31], "num_valu": 18, "pvalu": 18, "simpl": [18, 22, 29, 31], "spearman": [18, 30], "correl": [18, 30], "pair": 18, "onli": [18, 20, 22, 25, 26, 28, 34], "mani": [18, 20, 22, 30], "sort_values_arrai": 19, "sort_values_histori": 19, "map": 19, "sequenc": [19, 26, 30], "sort": 19, "sample_id": 19, "last": [19, 20, 22], "sort_valu": 19, "value_float": 19, "backward_elimin": 19, "indic": [19, 20, 22, 26, 28], "job_id": [19, 26, 31], "after": [19, 20, 22, 30], "increment": 19, "remov": 19, "duh": [19, 30], "split": [19, 21, 26, 31], "retrain": [19, 26, 27], "happen": 19, "posit": [19, 25, 26, 28, 31], "execut": [19, 26, 31], "forward_select": 19, "ad": 19, "addit": 19, "dure": 19, "select": 19, "backward": 19, "elimin": 19, "increas": [19, 20, 22, 28], "bootstrap_iter": [20, 22, 26, 35], "min_scor": [20, 22], "score_toler": [20, 22], "min_valu": [20, 22, 26, 30], "value_toler": [20, 22], "num_work": [20, 22], "run_id": [20, 22, 31], "approxim": [20, 22, 25, 30], "expect": [20, 22, 29], "we": [20, 22], "sequenti": [20, 22], "permut": [20, 22, 23], "stop": [20, 22, 26, 27, 30], "doesn": [20, 22], "t": [20, 22, 25, 26, 30, 31, 33], "beyond": [20, 22], "threshold": [20, 22], "keep": [20, 22], "sampl": [20, 22, 26, 28, 30], "updat": [20, 22, 27], "until": [20, 22, 26, 27], "chang": [20, 22], "move": [20, 22], "averag": [20, 22, 26, 27, 30, 31], "fall": [20, 22], "below": [20, 22, 26, 27], "anoth": [20, 22], "repeat": [20, 22, 26, 27, 31], "global_scor": [20, 22], "time": [20, 22, 26, 31], "estim": [20, 22, 30], "its": [20, 22], "varianc": [20, 22, 27], "stddev": [20, 22], "bootstrap": [20, 22], "complet": [20, 22], "least": [20, 22, 30], "deriv": [20, 22, 26, 30], "min_step": [20, 22], "ep": [20, 22, 30], "close": [20, 22, 30], "never": [20, 22], "more": [20, 22, 25, 26, 27], "than": [20, 21, 22, 26, 27], "iter": [20, 22, 25, 26, 27, 28, 30, 33], "total": [20, 22], "across": [20, 22, 26, 31], "worker": [20, 22, 31], "most": [20, 22], "g": [20, 22, 26, 30, 31, 34], "available_cpu": [20, 22, 26, 31], "thread": [20, 22, 26, 31], "purpos": [20, 22, 26, 31], "locat": [20, 22], "serial_truncated_montecarlo_shaplei": [20, 22], "truncat": [20, 22], "cpu": [20, 22], "permutation_montecarlo_shaplei": [20, 22], "combinatori": [20, 22, 23], "definit": [20, 22, 23, 25], "combinatorial_exact_shaplei": [20, 23], "exact": [20, 21, 23], "permutation_exact_shaplei": [20, 23], "classifi": 21, "regressor": 21, "extract": 21, "modifi": 21, "nor": 21, "other": [21, 30], "get_param": 21, "datashaplei": 22, "don": [22, 31, 33], "foolishli": 22, "rai": 22, "whatev": [22, 29], "distribut": [22, 27, 30], "multipl": [22, 25, 29, 31], "machin": 22, "provid": [22, 31], "singl": 22, "montecarlo_shaplei": 22, "backend": [22, 26, 27, 31], "argument": [22, 26, 27, 30, 31], "multiprocess": [22, 26, 31], "serial": 22, "group": [22, 28], "conjugate_gradi": 25, "union": [25, 26, 33], "callabl": [25, 26, 27, 30, 31], "x0": 25, "m": [25, 30], "rtol": 25, "1e": 25, "05": 25, "verify_assumpt": 25, "raise_except": [25, 29], "batch": 25, "algorithm": [25, 27, 31], "vector": [25, 30, 34], "product": [25, 34], "effici": 25, "see": [25, 26, 30, 31], "http": [25, 29, 30, 31], "en": 25, "wikipedia": 25, "org": [25, 29], "conjugate_gradient_method": 25, "detail": 25, "github": [25, 30, 31], "com": [25, 30, 31], "scipi": 25, "blob": 25, "v1": 25, "spars": 25, "linalg": 25, "_isolv": 25, "l282": 25, "l351": 25, "web": 25, "stanford": 25, "edu": 25, "ee364b": 25, "lectur": 25, "conj_grad_slid": 25, "pdf": 25, "f": [25, 30], "r": [25, 26, 31], "dimens": [25, 26, 28], "inv": 25, "underli": 25, "symmetr": 25, "maximum": 25, "rel": [25, 26, 27], "toler": [25, 26, 27, 30], "residu": 25, "norm": 25, "iff": 25, "stochast": [25, 30], "rule": 25, "assumpt": 25, "rais": 25, "warn": 25, "solut": 25, "conjugate_gradient_error_bound": 25, "xt": 25, "math": [25, 30], "stackexchang": [25, 30], "question": [25, 30], "382958": 25, "error": 25, "client_config": [26, 27, 30], "clientconfig": [26, 27, 30], "cache_threshold": [26, 27], "allow_repeated_train": [26, 27], "rtol_threshold": [26, 27], "min_repetit": [26, 27], "ignore_arg": [26, 27], "decor": [26, 27, 34], "have": [26, 27, 30], "transpar": [26, 27], "code": [26, 27], "constant": [26, 27], "except": [26, 27], "those": [26, 27, 28], "remot": [26, 27], "due": [26, 27], "pickl": [26, 27, 29], "drawback": [26, 27], "messi": [26, 27], "docstr": [26, 27], "config": [26, 27], "client": [26, 27, 30], "Will": [26, 27], "merg": [26, 27], "top": [26, 27], "configur": [26, 27, 29, 30, 35], "same": [26, 27, 31], "re": [26, 27], "precis": [26, 27], "reach": [26, 27], "minimum": [26, 27], "repetit": [26, 27], "do": [26, 27, 31, 33], "keyword": [26, 27], "account": [26, 27], "hash": [26, 27], "wrap": [26, 27, 33], "usag": [26, 27], "para": [26, 27], "onc": [26, 27], "smaller": [26, 27], "given": [26, 27, 28], "default_config": [26, 27], "server": [26, 27, 29], "localhost": [26, 27, 29], "connect_timeout": [26, 27], "timeout": [26, 27, 31], "packet": [26, 27], "consolid": [26, 27], "no_delai": [26, 27], "serd": [26, 27], "pickleserd": [26, 27], "pickle_vers": [26, 27], "arg": [26, 31, 34], "protocol": [26, 34], "pedant": [26, 34], "hint": [26, 34], "feature_nam": [26, 28], "target_nam": [26, 28], "data_nam": [26, 28], "descript": [26, 28], "better": [26, 28, 35], "handl": [26, 28, 29], "dval": [26, 28], "name": [26, 28, 31], "indexexpress": [26, 28], "get_train_data": [26, 28], "train_indic": [26, 28], "differ": [26, 27, 28], "sub": [26, 28, 31], "notic": [26, 28], "typic": [26, 28], "equal": [26, 28], "full": [26, 28], "subset": [26, 28, 30], "properti": [26, 28, 30, 33], "contigu": [26, 28], "integ": [26, 28, 30], "len": [26, 28], "individu": [26, 28], "datapoint": [26, 28], "classmethod": [26, 28], "bunch": [26, 28], "train_siz": [26, 28], "load_": [26, 28], "pd": [26, 28], "panda": [26, 28], "home": [26, 28], "runner": [26, 28], "tox": [26, 28], "doc": [26, 28, 29], "lib": [26, 28], "python3": [26, 28], "site": [26, 28], "packag": [26, 28], "__init__": [26, 28], "from_panda": [26, 28], "df": [26, 28], "datafram": [26, 28], "That": [26, 28, 30, 35], "mapreducejob": [26, 31], "collect": [26, 30, 31], "loki": [26, 31], "embarrassingli": [26, 31], "ot": [26, 31], "alloc": [26, 31], "90": [26, 31], "whole": [26, 31], "evenli": [26, 31], "among": [26, 31], "two": [26, 31], "core": [26, 31], "five": [26, 31], "success": [26, 31], "receiv": [26, 29, 31], "per": [26, 31], "reduc": [26, 31], "from_fun": [26, 31], "accept": [26, 30, 31], "etc": [26, 31], "kwd": [26, 31], "There": [26, 31], "probabl": [26, 30, 31], "77": [26, 31], "just": [26, 29, 31], "static": [26, 31], "lambda": [26, 31], "job_id_arg": [26, 31], "run_id_arg": [26, 31], "Not": [26, 31], "actual": [26, 31], "oper": [26, 31, 34], "reduct": [26, 31], "itself": [26, 31], "join": [26, 31], "functool": [26, 31], "id": [26, 31], "allow": [26, 27, 31, 34], "global": [26, 31], "support": [26, 31, 33], "vanishing_deriv": [26, 30], "atol": [26, 30], "row": [26, 30], "whose": [26, 30], "empir": [26, 30], "converg": [26, 30], "zero": [26, 30], "absolut": [26, 30], "unpack": [26, 34], "cl": [26, 31, 34], "attribut": [26, 34], "doubl": [26, 34], "asterisk": [26, 34], "dataclass": [26, 34], "schtuff": [26, 34], "meh": [26, 31, 34], "d": [26, 34], "scorer": [26, 35], "catch_error": [26, 35], "default_scor": [26, 35], "enable_cach": [26, 30, 35], "cache_opt": [26, 35], "memcachedconfig": [26, 27, 35], "conveni": [26, 35], "wrapper": [26, 35], "memoiz": [26, 35], "bootstrap_test_scor": [26, 35], "lack": [26, 35], "place": [26, 35], "powerset": [26, 30], "power": [26, 30], "grow": [26, 30], "size": [26, 30], "random_powerset": [26, 30], "random": [26, 30], "maybe_progress": [26, 33], "rang": [26, 33], "tqdm_kwarg": [26, 33], "tqdm_asyncio": [26, 33], "mock": [26, 33], "well": [26, 33], "ani": [26, 30, 31, 33], "access": [26, 33], "todo": [27, 31], "0x7f72d7403a00": 27, "factori": 27, "get_running_avg_vari": 27, "previous_avg": 27, "previous_vari": 27, "new_valu": 27, "count": 27, "welford": 27, "previou": 27, "step": [27, 30], "new": 27, "seri": 27, "seen": 27, "far": 27, "new_averag": 27, "new_vari": 27, "groupeddataset": 28, "data_group": 28, "coalit": 28, "from_dataset": 28, "polynomi": 28, "coeffici": 28, "polynomial_dataset": 28, "must": [28, 30], "monomi": 28, "degre": 28, "flip_dataset": 28, "flip_percentag": 28, "in_plac": 28, "binari": 28, "classif": 28, "problem": 28, "invert": 28, "percentag": 28, "label": 28, "between": 28, "describ": 28, "how": 28, "much": 28, "shall": 28, "flip": 28, "old": 28, "orign": 28, "mostli": 29, "quick": 29, "hack": 29, "debug": 29, "logrecordstreamhandl": 29, "request": 29, "client_address": 29, "streamrequesthandl": 29, "handler": 29, "stream": 29, "basic": 29, "record": 29, "polici": 29, "local": 29, "cookbook": 29, "howto": 29, "html": 29, "byte": 29, "length": 29, "logrecord": 29, "format": 29, "accord": 29, "handle_log_record": 29, "logrecordsocketreceiv": 29, "host": 29, "port": 29, "threadingtcpserv": 29, "tcp": 29, "socket": 29, "suitabl": 29, "almost": 29, "verbatim": 29, "allow_reuse_address": 29, "serve_until_stop": 29, "start_logging_serv": 29, "9020": 29, "set_logg": 29, "_logger": 29, "raise_or_log": 29, "messag": 29, "mcmc_is_linear_funct": 30, "verify_sampl": 30, "1000": 30, "noth": 30, "sum_i": 30, "a_i": 30, "v_i": 30, "mcmc_is_linear_function_positive_definit": 30, "lower_bound_hoeffd": 30, "delta": 30, "score_rang": 30, "lower": 30, "bound": 30, "obtain": 30, "\u03b5": 30, "\u03b4": 30, "quantiti": 30, "taken": 30, "powersetdistribut": 30, "uniform": 30, "weight": 30, "max_subset": 30, "dist": 30, "uniformli": 30, "without": 30, "pre": 30, "arbitrarili": 30, "larg": 30, "howev": 30, "ten": 30, "thousand": 30, "veri": 30, "long": 30, "henc": 30, "abil": 30, "wish": 30, "determinist": 30, "empti": 30, "like": 30, "distinct": 30, "rank": 30, "revers": 30, "perfect": 30, "match": 30, "independ": 30, "random_matrix_with_condition_numb": 30, "condition_numb": 30, "positive_definit": 30, "gist": 30, "bstellato": 30, "23322fe5d87bb71da922fbc41d658079": 30, "file": 30, "random_mat_condition_numb": 30, "1351616": 30, "condit": 30, "ata": 30, "linear_regression_analytical_derivative_d_theta": 30, "bxm": 30, "nparrai": 30, "bxn": 30, "bx": 30, "where": 30, "d_theta": 30, "l": 30, "d_b": 30, "linear_regression_analytical_derivative_d2_theta": 30, "linear_regression_analytical_derivative_d_x_d_theta": 30, "xm": 30, "sample_classification_dataset_using_gaussian": 30, "mu": 30, "sigma": 30, "num_sampl": 30, "decision_boundary_fixed_variance_2d": 30, "mu_1": 30, "mu_2": 30, "outdat": 31, "comment": 31, "some": 31, "statu": 31, "histor": 31, "inform": 31, "gather": 31, "later": 31, "ident": 31, "make_nested_backend": 31, "nest": 31, "would": 31, "sequentialbackend": 31, "issu": 31, "947": 31, "chunkifi": 31, "njob": 31, "interruptiblework": 31, "worker_id": 31, "queue": 31, "abort": 31, "consum": 31, "_run": 31, "self": 31, "shapleywork": 31, "instanti": 31, "coordin": 31, "therein": 31, "share": 31, "avoid": 31, "both": 31, "flag": 31, "processor": 31, "put": 31, "get_and_process": 31, "clear_task": 31, "clear_result": 31, "pbar": 31, "end": 31, "plot_dataset": 32, "x_min": 32, "x_max": 32, "line": 32, "mockprogress": 33, "anyth": 33, "minimock": 33, "input": 34, "loss": 34}, "objects": {"": [[6, 0, 0, "-", "valuation"]], "valuation": [[5, 0, 0, "-", "cli"], [7, 0, 0, "-", "influence"], [11, 0, 0, "-", "loo"], [13, 0, 0, "-", "models"], [17, 0, 0, "-", "reporting"], [20, 0, 0, "-", "shapley"], [24, 0, 0, "-", "solve"], [26, 0, 0, "-", "utils"]], "valuation.cli": [[5, 1, 1, "", "maybe_init_task"]], "valuation.influence": [[8, 0, 0, "-", "general"], [9, 0, 0, "-", "linear"], [10, 0, 0, "-", "types"]], "valuation.influence.general": [[8, 1, 1, "", "influences"]], "valuation.influence.linear": [[9, 1, 1, "", "influences_perturbation_linear_regression_analytical"], [9, 1, 1, "", "influences_up_linear_regression_analytical"], [9, 1, 1, "", "linear_influences"]], "valuation.influence.types": [[10, 2, 1, "", "InfluenceTypes"]], "valuation.influence.types.InfluenceTypes": [[10, 3, 1, "", "Perturbation"], [10, 3, 1, "", "Up"]], "valuation.loo": [[12, 0, 0, "-", "naive"]], "valuation.loo.naive": [[12, 1, 1, "", "naive_loo"]], "valuation.models": [[14, 0, 0, "-", "binary_logistic_regression"], [15, 0, 0, "-", "linear_regression_torch_model"], [16, 0, 0, "-", "pytorch_model"]], "valuation.models.binary_logistic_regression": [[14, 2, 1, "", "BinaryLogisticRegressionTorchModel"]], "valuation.models.binary_logistic_regression.BinaryLogisticRegressionTorchModel": [[14, 4, 1, "", "forward"], [14, 3, 1, "", "training"]], "valuation.models.linear_regression_torch_model": [[15, 2, 1, "", "LRTorchModel"]], "valuation.models.linear_regression_torch_model.LRTorchModel": [[15, 4, 1, "", "forward"], [15, 3, 1, "", "training"]], "valuation.models.pytorch_model": [[16, 2, 1, "", "PyTorchOptimizer"], [16, 2, 1, "", "PyTorchSupervisedModel"], [16, 1, 1, "", "flatten_gradient"], [16, 1, 1, "", "tt"]], "valuation.models.pytorch_model.PyTorchOptimizer": [[16, 3, 1, "", "ADAM"], [16, 3, 1, "", "ADAM_W"]], "valuation.models.pytorch_model.PyTorchSupervisedModel": [[16, 4, 1, "", "fit"], [16, 4, 1, "", "grad"], [16, 4, 1, "", "mvp"], [16, 4, 1, "", "num_params"], [16, 4, 1, "", "predict"], [16, 4, 1, "", "score"]], "valuation.reporting": [[18, 0, 0, "-", "plots"], [19, 0, 0, "-", "scores"]], "valuation.reporting.plots": [[18, 1, 1, "", "shaded_mean_std"], [18, 1, 1, "", "shapley_results"], [18, 1, 1, "", "spearman_correlation"]], "valuation.reporting.scores": [[19, 1, 1, "", "backward_elimination"], [19, 1, 1, "", "compute_fb_scores"], [19, 1, 1, "", "forward_selection"], [19, 1, 1, "", "sort_values"], [19, 1, 1, "", "sort_values_array"], [19, 1, 1, "", "sort_values_history"]], "valuation.shapley": [[20, 1, 1, "", "combinatorial_exact_shapley"], [20, 1, 1, "", "combinatorial_montecarlo_shapley"], [21, 0, 0, "-", "knn"], [22, 0, 0, "-", "montecarlo"], [23, 0, 0, "-", "naive"], [20, 1, 1, "", "permutation_exact_shapley"], [20, 1, 1, "", "permutation_montecarlo_shapley"], [20, 1, 1, "", "serial_truncated_montecarlo_shapley"], [20, 1, 1, "", "truncated_montecarlo_shapley"]], "valuation.shapley.knn": [[21, 1, 1, "", "exact_knn_shapley"]], "valuation.shapley.montecarlo": [[22, 1, 1, "", "combinatorial_montecarlo_shapley"], [22, 1, 1, "", "permutation_montecarlo_shapley"], [22, 1, 1, "", "serial_truncated_montecarlo_shapley"], [22, 1, 1, "", "truncated_montecarlo_shapley"]], "valuation.shapley.naive": [[23, 1, 1, "", "combinatorial_exact_shapley"], [23, 1, 1, "", "permutation_exact_shapley"]], "valuation.solve": [[25, 0, 0, "-", "cg"]], "valuation.solve.cg": [[25, 1, 1, "", "conjugate_gradient"], [25, 1, 1, "", "conjugate_gradient_error_bound"]], "valuation.utils": [[26, 2, 1, "", "Dataset"], [26, 2, 1, "", "MapReduceJob"], [26, 2, 1, "", "SupervisedModel"], [26, 2, 1, "", "Utility"], [26, 1, 1, "", "available_cpus"], [26, 1, 1, "", "bootstrap_test_score"], [27, 0, 0, "-", "caching"], [28, 0, 0, "-", "dataset"], [29, 0, 0, "-", "logging"], [26, 1, 1, "", "map_reduce"], [26, 1, 1, "", "maybe_progress"], [26, 1, 1, "", "memcached"], [30, 0, 0, "-", "numeric"], [31, 0, 0, "-", "parallel"], [32, 0, 0, "-", "plotting"], [26, 1, 1, "", "powerset"], [33, 0, 0, "-", "progress"], [34, 0, 0, "-", "types"], [26, 1, 1, "", "unpackable"], [35, 0, 0, "-", "utility"], [26, 1, 1, "", "vanishing_derivatives"]], "valuation.utils.Dataset": [[26, 5, 1, "", "data_names"], [26, 5, 1, "", "dim"], [26, 4, 1, "", "feature"], [26, 4, 1, "", "from_pandas"], [26, 4, 1, "", "from_sklearn"], [26, 4, 1, "", "get_train_data"], [26, 5, 1, "", "indices"], [26, 3, 1, "", "pd"], [26, 4, 1, "", "target"]], "valuation.utils.MapReduceJob": [[26, 4, 1, "", "from_fun"], [26, 4, 1, "", "reduce"]], "valuation.utils.SupervisedModel": [[26, 4, 1, "", "fit"], [26, 4, 1, "", "predict"], [26, 4, 1, "", "score"]], "valuation.utils.Utility": [[26, 3, 1, "", "data"], [26, 3, 1, "", "model"], [26, 3, 1, "", "scoring"]], "valuation.utils.caching": [[27, 2, 1, "", "ClientConfig"], [27, 2, 1, "", "MemcachedConfig"], [27, 1, 1, "", "get_running_avg_variance"], [27, 1, 1, "", "memcached"]], "valuation.utils.caching.ClientConfig": [[27, 3, 1, "", "connect_timeout"], [27, 4, 1, "", "items"], [27, 4, 1, "", "keys"], [27, 3, 1, "", "no_delay"], [27, 3, 1, "", "serde"], [27, 3, 1, "", "server"], [27, 3, 1, "", "timeout"], [27, 4, 1, "", "update"]], "valuation.utils.caching.MemcachedConfig": [[27, 3, 1, "", "allow_repeated_training"], [27, 3, 1, "", "cache_threshold"], [27, 3, 1, "", "client_config"], [27, 3, 1, "", "ignore_args"], [27, 4, 1, "", "items"], [27, 4, 1, "", "keys"], [27, 3, 1, "", "min_repetitions"], [27, 3, 1, "", "rtol_threshold"], [27, 4, 1, "", "update"]], "valuation.utils.dataset": [[28, 2, 1, "", "Dataset"], [28, 2, 1, "", "GroupedDataset"], [28, 1, 1, "", "flip_dataset"], [28, 1, 1, "", "polynomial"], [28, 1, 1, "", "polynomial_dataset"]], "valuation.utils.dataset.Dataset": [[28, 5, 1, "", "data_names"], [28, 5, 1, "", "dim"], [28, 4, 1, "", "feature"], [28, 4, 1, "", "from_pandas"], [28, 4, 1, "", "from_sklearn"], [28, 4, 1, "", "get_train_data"], [28, 5, 1, "", "indices"], [28, 3, 1, "", "pd"], [28, 4, 1, "", "target"]], "valuation.utils.dataset.GroupedDataset": [[28, 5, 1, "", "data_names"], [28, 4, 1, "", "from_dataset"], [28, 4, 1, "", "from_sklearn"], [28, 4, 1, "", "get_train_data"], [28, 5, 1, "", "indices"]], "valuation.utils.logging": [[29, 2, 1, "", "LogRecordSocketReceiver"], [29, 2, 1, "", "LogRecordStreamHandler"], [29, 1, 1, "", "raise_or_log"], [29, 1, 1, "", "set_logger"], [29, 1, 1, "", "start_logging_server"]], "valuation.utils.logging.LogRecordSocketReceiver": [[29, 3, 1, "", "allow_reuse_address"], [29, 4, 1, "", "serve_until_stopped"]], "valuation.utils.logging.LogRecordStreamHandler": [[29, 4, 1, "", "handle"], [29, 4, 1, "", "handle_log_record"]], "valuation.utils.numeric": [[30, 2, 1, "", "PowerSetDistribution"], [30, 1, 1, "", "decision_boundary_fixed_variance_2d"], [30, 1, 1, "", "linear_regression_analytical_derivative_d2_theta"], [30, 1, 1, "", "linear_regression_analytical_derivative_d_theta"], [30, 1, 1, "", "linear_regression_analytical_derivative_d_x_d_theta"], [30, 1, 1, "", "lower_bound_hoeffding"], [30, 1, 1, "", "mcmc_is_linear_function"], [30, 1, 1, "", "mcmc_is_linear_function_positive_definite"], [30, 1, 1, "", "powerset"], [30, 1, 1, "", "random_matrix_with_condition_number"], [30, 1, 1, "", "random_powerset"], [30, 1, 1, "", "sample_classification_dataset_using_gaussians"], [30, 1, 1, "", "spearman"], [30, 1, 1, "", "vanishing_derivatives"]], "valuation.utils.numeric.PowerSetDistribution": [[30, 3, 1, "", "UNIFORM"], [30, 3, 1, "", "WEIGHTED"]], "valuation.utils.parallel": [[31, 2, 1, "", "Coordinator"], [31, 1, 1, "", "Identity"], [31, 2, 1, "", "InterruptibleWorker"], [31, 2, 1, "", "MapReduceJob"], [31, 1, 1, "", "available_cpus"], [31, 1, 1, "", "chunkify"], [31, 1, 1, "", "make_nested_backend"], [31, 1, 1, "", "map_reduce"]], "valuation.utils.parallel.Coordinator": [[31, 4, 1, "", "clear_results"], [31, 4, 1, "", "clear_tasks"], [31, 4, 1, "", "end"], [31, 4, 1, "", "get_and_process"], [31, 4, 1, "", "instantiate"], [31, 4, 1, "", "put"], [31, 4, 1, "", "start"]], "valuation.utils.parallel.InterruptibleWorker": [[31, 4, 1, "", "aborted"], [31, 4, 1, "", "run"]], "valuation.utils.parallel.MapReduceJob": [[31, 4, 1, "", "from_fun"], [31, 4, 1, "", "reduce"]], "valuation.utils.plotting": [[32, 1, 1, "", "plot_datasets"]], "valuation.utils.progress": [[33, 2, 1, "", "MockProgress"], [33, 1, 1, "", "maybe_progress"]], "valuation.utils.progress.MockProgress": [[33, 2, 1, "", "MiniMock"]], "valuation.utils.types": [[34, 2, 1, "", "SupervisedModel"], [34, 2, 1, "", "TorchObjective"], [34, 2, 1, "", "TwiceDifferentiable"], [34, 1, 1, "", "unpackable"]], "valuation.utils.types.SupervisedModel": [[34, 4, 1, "", "fit"], [34, 4, 1, "", "predict"], [34, 4, 1, "", "score"]], "valuation.utils.types.TwiceDifferentiable": [[34, 4, 1, "", "grad"], [34, 4, 1, "", "mvp"], [34, 4, 1, "", "num_params"]], "valuation.utils.utility": [[35, 2, 1, "", "Utility"], [35, 1, 1, "", "bootstrap_test_score"]], "valuation.utils.utility.Utility": [[35, 3, 1, "", "data"], [35, 3, 1, "", "model"], [35, 3, 1, "", "scoring"]]}, "objtypes": {"0": "py:module", "1": "py:function", "2": "py:class", "3": "py:attribute", "4": "py:method", "5": "py:property"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"], "2": ["py", "class", "Python class"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "method", "Python method"], "5": ["py", "property", "Python property"]}, "titleterms": {"get": 0, "start": 0, "creat": 0, "dataset": [0, 28], "util": [0, 26, 35], "comput": 0, "shaplei": [0, 4, 20], "valu": 0, "pydvl": [1, 2], "guid": 1, "tutori": 1, "modul": 1, "indic": 1, "tabl": 1, "instal": 2, "depend": 2, "notebook": 3, "knn": [4, 21], "cli": 5, "valuat": 6, "influenc": 7, "gener": 8, "linear": 9, "type": [10, 34], "loo": 11, "naiv": [12, 23], "model": 13, "binary_logistic_regress": 14, "linear_regression_torch_model": 15, "pytorch_model": 16, "report": 17, "plot": [18, 32], "score": 19, "montecarlo": 22, "todo": 22, "solv": 24, "cg": 25, "cach": 27, "log": 29, "numer": 30, "parallel": 31, "progress": 33}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.todo": 2, "nbsphinx": 4, "sphinx": 56}})