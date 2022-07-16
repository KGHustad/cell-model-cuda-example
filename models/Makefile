models = $(wildcard *.ode)
models_cuda = $(patsubst %.ode,%.cu,$(models))


solver_args = --solvers.explicit_euler.generate=1 --solvers.explicit_euler.function_name=step_FE --solvers.generalized_rush_larsen.generate=1 --solvers.generalized_rush_larsen.function_name=step_GRL1 --code.body.use_enum=1
solver_args += --solvers.explicit_euler.function_name=step_FE
solver_args += --solvers.generalized_rush_larsen.function_name=step_GRL1
solver_args += --functions.monitored.generate=0 --functions.rhs.generate=0

gotran2cuda_args = $(solver_args)

all: $(models_cuda)

%.cu : %.ode
	python -m gotran gotran2cuda $< $(gotran2cuda_args) --output=$@
