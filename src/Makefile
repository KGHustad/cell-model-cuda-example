# Rules to update the auto-generated command-line argument parser by running gengetopt
cmdline_files_basename = cli

all: $(cmdline_files_basename).c

$(cmdline_files_basename).c $(cmdline_files_basename).h: cli.ggo
	gengetopt -l -C -F $(cmdline_files_basename) < $<
