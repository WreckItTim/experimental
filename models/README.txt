
Directory breakdown:

    ~/model_name/modeling/model_XXX.zip    >>    trained model at given label XXX
    ~/model_name/test_XXX/    >>    test folder resulting from given label XXX (i.e. evlaluating a trained model)
    ~/model_name/test_XXX/states    >>    json files containing episode data and each state from given test
    ~/model_name/test_XXX/configuration.json    >>    configuration json file to be read by DRL repo
    ~/model_name/test_XXX/evaluation.json    >>    evaluation json file with summary of results over all test paths


Available models:

    navigation_airsim_blocks_dqn_2d    >>    navigation model with horizontal motion only, trained on the AirSim Blocks map with a DQN, inputs GPS coodinates of drone and target position and a forward facing ground truth depth map