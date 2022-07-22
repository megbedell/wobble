
Mojave Installation
+++++++++++++++++++

If you are seeing errors like this during the build process in macOS
10.14 (Mojave)::
  ld: library not found for -lstdc++
  clang: error: linker command failed with exit code 1 (use -v to see invocation)
  error: command 'g++' failed with exit status 1
try the solution suggested `here <https://blog.driftingruby.com/updated-to-mojave/>`_

In summary, you might need to install the macOS developer SDK header
files separately in Mojave. Open the file
``/Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg``
and follow the usual installer instructions.

Then try to build `wobble` again.
