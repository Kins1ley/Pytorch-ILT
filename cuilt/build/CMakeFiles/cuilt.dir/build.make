# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhubinwu/L2ILT/cuilt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhubinwu/L2ILT/cuilt/build

# Include any dependencies generated for this target.
include CMakeFiles/cuilt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuilt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuilt.dir/flags.make

CMakeFiles/cuilt.dir/main.cpp.o: CMakeFiles/cuilt.dir/flags.make
CMakeFiles/cuilt.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhubinwu/L2ILT/cuilt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuilt.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuilt.dir/main.cpp.o -c /home/zhubinwu/L2ILT/cuilt/main.cpp

CMakeFiles/cuilt.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuilt.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhubinwu/L2ILT/cuilt/main.cpp > CMakeFiles/cuilt.dir/main.cpp.i

CMakeFiles/cuilt.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuilt.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhubinwu/L2ILT/cuilt/main.cpp -o CMakeFiles/cuilt.dir/main.cpp.s

# Object files for target cuilt
cuilt_OBJECTS = \
"CMakeFiles/cuilt.dir/main.cpp.o"

# External object files for target cuilt
cuilt_EXTERNAL_OBJECTS =

cuilt: CMakeFiles/cuilt.dir/main.cpp.o
cuilt: CMakeFiles/cuilt.dir/build.make
cuilt: liblibcuilt.so
cuilt: libiltproto.a
cuilt: /usr/lib/x86_64-linux-gnu/libgflags.so
cuilt: /usr/lib/x86_64-linux-gnu/libglog.so
cuilt: /usr/lib/x86_64-linux-gnu/libprotobuf.so
cuilt: CMakeFiles/cuilt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhubinwu/L2ILT/cuilt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cuilt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuilt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuilt.dir/build: cuilt

.PHONY : CMakeFiles/cuilt.dir/build

CMakeFiles/cuilt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuilt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuilt.dir/clean

CMakeFiles/cuilt.dir/depend:
	cd /home/zhubinwu/L2ILT/cuilt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhubinwu/L2ILT/cuilt /home/zhubinwu/L2ILT/cuilt /home/zhubinwu/L2ILT/cuilt/build /home/zhubinwu/L2ILT/cuilt/build /home/zhubinwu/L2ILT/cuilt/build/CMakeFiles/cuilt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuilt.dir/depend

