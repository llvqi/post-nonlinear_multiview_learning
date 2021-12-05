#!/bin/bash

# Generate synthetic data
python generate_synthetic_data.py

# Run the algorithm
python post-nonlinear_multiview_learning_demo.py \
--num_views=2 \
--latent_dim=2 \
--batch_size=1000 \
--num_epochs=100 \
--inner_iters=100 \
--learning_rate=1e-3 \
--_lambda=1e-3 \
--model_file="best_model_multiview.pth" \
--f_num_layers=1 \
--f_hidden_size=128 \
--g_num_layers=1 \
--g_hidden_size=128 \

