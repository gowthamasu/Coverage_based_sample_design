
# Coverage-Based Designs Improve Sample Mining and Hyper-Parameter Optimization

### Abstract : 
Sampling one or more effective solutions from large search spaces is a recurring idea in machine learning, and sequential optimization has become a popular solution. Typical examples include data summarization, sample mining for predictive modeling and hyper-parameter optimization. Existing solutions attempt to adaptively trade-off between global exploration and local exploitation, wherein the initial exploratory sample is critical to their success. While discrepancy-based samples have become the de facto approach for exploration, results from computer graphics suggest that coverage-based designs, e.g. Poisson disk sampling, can be a superior alternative. In order to successfully adopt coverage-based sample designs to machine learning applications, which were originally developed for 2-d image analysis, we propose fundamental advances by constructing a parameterized family of designs with provably improved coverage characteristics, and by developing algorithms for effective sample synthesis. Using experiments in sample mining and hyper-parameter optimization for supervised learning, we show that our approach consistently outperforms existing exploratory sampling methods in both blind exploration, and sequential search with Bayesian optimization.

## Codes Description : 


#### --- SFSD.py :
        This code generate space filling spectral designs (SFSD). 
	SFSD.__init__            :: defaults 
	SFSD.choose_sigma	 :: snippet to choose sigma based on dimension
	SFSD.edge_correction     :: Computing the edge correction factor
	SFSD.G_kern              :: Faster Gaussian kernel computation
	SFSD.r_min               :: r_min for step design
	SFSD.initial_calculation :: defining PCF and initial calculation
	SFSD.generate		 :: Sample generation


#### --- optimal_params.py :
	optimal_params.__init__             :: defaults
	optimal_params.rmin                 :: initial r0 by r_min of step design
	optimal_params.r_1                  :: initial r1 >> r0
	optimal_params.PSD                  :: Compute power spectral density
	optimal_params.compute_params       :: optimization procedure to find optimal r0 and r1


#### --- blind_exploration.py :
	mnist_hypopt                        :: blind exploration code for MNIST dataset
		run_blindexploration        :: Build a CNN model and pass the set of hyperparameters to be searched
		scale_points                :: scale the search space   
		start_exploration	    :: start exploration for every sample loaded from sample design


#### --- sequential_sampling.py :
	bayesian_opt		      :: bayesian optimization pipeline with your choice of initial exploratory sample design
		CNN_model             :: Build the CNN model
		f(x)                  :: function to optimize CNN model
		scale                 :: change scale of search space
		
