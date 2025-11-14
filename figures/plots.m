function plot_model_accuracies(model_names, train_acc, val_acc, test_acc, output_filename)
%PLOT_MODEL_ACCURACIES  Create a clustered bar chart of model accuracies.
%
%   plot_model_accuracies(model_names, train_acc, val_acc, test_acc, output_filename)
%
%   Inputs:
%       model_names     - cell array of strings, e.g. {'Model1','Model2','Model3'}
%       train_acc       - numeric array of training accuracies (same length as model_names)
%       val_acc         - numeric array of validation accuracies
%       test_acc        - numeric array of test accuracies
%       output_filename - string for saving the figure, e.g. 'model_performance.fig'
%
%   Example:
%       models = {'ByChar', 'GRU', 'Transformer'};
%       train = [100, 99.5, 98.7];
%       val   = [100, 67.3, 85.4];
%       test  = [100, 62.8, 83.9];
%       plot_model_accuracies(models, train, val, test, 'performance.fig')

    % Validate input lengths
    n = numel(model_names);
    if any([numel(train_acc), numel(val_acc), numel(test_acc)] ~= n)
        error('All input arrays must have the same length as model_names.');
    end

    % Combine data into matrix for bar plot
    data = [train_acc(:), val_acc(:), test_acc(:)];

    % Create figure
    figure('Color','w');
    b = bar(data, 'grouped');
    hold on;

    % Aesthetics
    title('Model Performance on Known-Plaintext Attack Task', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Accuracy (%)', 'FontSize', 12);
    xlabel('Model', 'FontSize', 12);
    set(gca, 'XTickLabel', model_names, 'XTick', 1:n, 'FontSize', 11);
    ylim([0 1]);
    grid on;

    % Legend
    legend({'Training', 'Validation', 'Test'}, 'Location', 'northoutside', 'Orientation', 'horizontal');

    % Add text labels above bars
    for i = 1:numel(b)
        xtips = b(i).XEndPoints;
        ytips = b(i).YEndPoints;
        labels = string(round(b(i).YData,1));
        text(xtips, ytips + 2, labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',9);
    end

    hold off;

    % Save figure
    if nargin > 4 && ~isempty(output_filename)
        savefig(output_filename);
        fprintf('Figure saved as %s\n', output_filename);
    end
end


model_names = { ...
    'Simple Autoencoder', ...              % AutoEncoderSimple_model
    ... % 'One-Hot Autoencoder', ...             % AutoEncoderOneHot_model
    'Embedded Autoencoder', ...            % AutoEncoderEmbedding_model
    ... % 'Reduced Embedding Autoencoder', ...   % AutoEncoderEmbeddingReduced_model
    'Reduced Embedding (By-Character)', ...% AutoEncoderEmbeddingReducedByChar_model
    'Embedded (By-Character) Autoencoder'  % AutoEncoderEmbeddingByChar_model
};

train_acc = [0.263824014400494, 0.9991262473968086, 0.9627406925639055, 1.0];
val_acc   = [0.12459751913433624, 0.6136711533386118, 0.9623383478490367, 1.0];
test_acc  = [0.12282017626101631, 0.6204762797674854, 0.962363719161019, 1.0];

plot_model_accuracies(model_names, train_acc, val_acc, test_acc, 'AE_substitution_model_performance.fig');

train_acc = [0.16754041926985816, 0.9990178495174652, 1.0, 1.0];
val_acc   = [0.13689627870150436, 0.5512272367379256, 1.0, 1.0];
test_acc  = [0.13610672095577403, 0.5523291634298573, 1.0, 1.0];

plot_model_accuracies(model_names, train_acc, val_acc, test_acc, 'AE_transposition_model_performance.fig');

train_acc = [0.27570179414914237, 0.999931019531327, 1.0, 1.0];
val_acc   = [0.09685932963842703, 0.6196093956188968, 1.0, 1.0];
test_acc  = [0.09670247247595831, 0.6179582652487209, 1.0, 1.0];

plot_model_accuracies(model_names, train_acc, val_acc, test_acc, 'AE_product_model_performance.fig');

