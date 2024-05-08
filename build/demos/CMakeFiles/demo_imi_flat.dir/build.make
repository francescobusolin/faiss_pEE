# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/busolin/biEffortFaiss/experiments/faiss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/busolin/biEffortFaiss/experiments/faiss/build

# Include any dependencies generated for this target.
include demos/CMakeFiles/demo_imi_flat.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include demos/CMakeFiles/demo_imi_flat.dir/compiler_depend.make

# Include the progress variables for this target.
include demos/CMakeFiles/demo_imi_flat.dir/progress.make

# Include the compile flags for this target's objects.
include demos/CMakeFiles/demo_imi_flat.dir/flags.make

demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o: demos/CMakeFiles/demo_imi_flat.dir/flags.make
demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o: /home/busolin/biEffortFaiss/experiments/faiss/demos/demo_imi_flat.cpp
demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o: demos/CMakeFiles/demo_imi_flat.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/busolin/biEffortFaiss/experiments/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o"
	cd /home/busolin/biEffortFaiss/experiments/faiss/build/demos && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o -MF CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o.d -o CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o -c /home/busolin/biEffortFaiss/experiments/faiss/demos/demo_imi_flat.cpp

demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.i"
	cd /home/busolin/biEffortFaiss/experiments/faiss/build/demos && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/busolin/biEffortFaiss/experiments/faiss/demos/demo_imi_flat.cpp > CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.i

demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.s"
	cd /home/busolin/biEffortFaiss/experiments/faiss/build/demos && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/busolin/biEffortFaiss/experiments/faiss/demos/demo_imi_flat.cpp -o CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.s

# Object files for target demo_imi_flat
demo_imi_flat_OBJECTS = \
"CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o"

# External object files for target demo_imi_flat
demo_imi_flat_EXTERNAL_OBJECTS =

demos/demo_imi_flat: demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o
demos/demo_imi_flat: demos/CMakeFiles/demo_imi_flat.dir/build.make
demos/demo_imi_flat: faiss/libfaiss.a
demos/demo_imi_flat: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
demos/demo_imi_flat: /usr/lib/x86_64-linux-gnu/libpthread.so
demos/demo_imi_flat: /home/busolin/biEffortFaiss/experiments/openBLAS/libopenblas.a
demos/demo_imi_flat: demos/CMakeFiles/demo_imi_flat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/busolin/biEffortFaiss/experiments/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_imi_flat"
	cd /home/busolin/biEffortFaiss/experiments/faiss/build/demos && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_imi_flat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demos/CMakeFiles/demo_imi_flat.dir/build: demos/demo_imi_flat
.PHONY : demos/CMakeFiles/demo_imi_flat.dir/build

demos/CMakeFiles/demo_imi_flat.dir/clean:
	cd /home/busolin/biEffortFaiss/experiments/faiss/build/demos && $(CMAKE_COMMAND) -P CMakeFiles/demo_imi_flat.dir/cmake_clean.cmake
.PHONY : demos/CMakeFiles/demo_imi_flat.dir/clean

demos/CMakeFiles/demo_imi_flat.dir/depend:
	cd /home/busolin/biEffortFaiss/experiments/faiss/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/busolin/biEffortFaiss/experiments/faiss /home/busolin/biEffortFaiss/experiments/faiss/demos /home/busolin/biEffortFaiss/experiments/faiss/build /home/busolin/biEffortFaiss/experiments/faiss/build/demos /home/busolin/biEffortFaiss/experiments/faiss/build/demos/CMakeFiles/demo_imi_flat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demos/CMakeFiles/demo_imi_flat.dir/depend

