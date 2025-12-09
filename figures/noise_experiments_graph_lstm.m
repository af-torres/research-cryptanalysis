% Data (as provided)
models = {'LSTM'};
noise = [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];

% === ENGLISH SENTENCES ===

% === (testing accuracies, truncated) ===
acc_sub = [ ...
    0.464  ... % noise = 0.0
    0.491  ... % noise = 0.1
    0.433  ... % noise = 0.2
    0.326  ... % noise = 0.3
    0.236  ... % noise = 0.4
    0.226  ... % noise = 0.5
    0.209  ... % noise = 0.6
    0.197  ... % noise = 0.7
    0.197  ... % noise = 0.8
];

acc_trans = [ ...
    0.576  ... % noise = 0.0
    0.554  ... % noise = 0.1
    0.377  ... % noise = 0.2
    0.401  ... % noise = 0.3
    0.301  ... % noise = 0.4
    0.241  ... % noise = 0.5
    0.214  ... % noise = 0.6
    0.199  ... % noise = 0.7
    0.195  ... % noise = 0.8
];

acc_prod = [ ...
    0.517  ... % noise = 0.0
    0.445  ... % noise = 0.1
    0.215  ... % noise = 0.2
    0.315  ... % noise = 0.3
    0.257  ... % noise = 0.4
    0.214  ... % noise = 0.5
    0.208  ... % noise = 0.6
    0.202  ... % noise = 0.7
    0.196  ... % noise = 0.8
];

% Create the figure
figure('Color','white');
hold on; grid on;

% Plot LSTM (solid lines)
p1 = plot(noise, acc_sub, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
p2 = plot(noise, acc_trans, '-s', 'LineWidth', 1.8, 'MarkerSize', 6);
p3 = plot(noise, acc_prod, '-^', 'LineWidth', 1.8, 'MarkerSize', 6);

% Axis labels
xlabel('Noise Level', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Test Accuracy', 'FontSize', 12, 'FontWeight', 'bold');

% Axis limits
xlim([0 0.8]);
ylim([0 1.05]);

% Title (optional for paper figures)
% title('Effect of Noise on Test Accuracy', 'FontSize', 13, 'FontWeight', 'bold');

% Legend (compact and clear)
legend([p1 p2 p3], ...
{'LSTM - Substitution', ...
 'LSTM - Transposition', ...
 'LSTM - Product', ...
}, ...
 'Location', 'southwest', 'FontSize', 10, 'Box', 'off');

% Publication formatting
set(gca, 'FontName', 'Times', 'FontSize', 11, 'LineWidth', 1);
set(gcf, 'Position', [100 100 600 400]);

hold off;

savefig("noise_experiments_english_lstm.fig");