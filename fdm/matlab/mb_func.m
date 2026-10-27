function row = mb_func(x)
% Callback handle for external MATLAB solvers (e.g. ORBITc's `func`
% argument). Evaluates the objective and all constraints at x through the
% live Python Oracle set up by fdm.fdm.matlab_bridge.mb_init, so every
% evaluation is counted/traced exactly like every other solver in this repo.
row = double(py.fdm.fdm.matlab_bridge.mb_eval_row(x(:)'));
end
