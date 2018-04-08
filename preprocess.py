#!/usr/bin/python
import sys
import time

def preprocess(ifile_path, ofile_path):
    start = time.time()
    with open(ifile_path) as ifile:
        nlines = 0
        with open(ofile_path, "w+") as ofile:
            for iline in ifile:
                nlines += 1
                iwords = iline.rstrip().split(" ")
                if (len(iwords) + 2 <= 30):
                    padding = " <pad>" * (30 - (len(iwords) + 2))
                    oline = "<bos> {0}{1} <eos>\n".format(" ".join(iwords), padding)
                    ofile.write(oline)
        end = time.time()
        print "Time Elapsed: {0} secs".format(end - start)

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print "Usage:   python preprocess.py <absolute_path>"
        sys.exit(1)
    ifile = sys.argv[1]
    ifile_name = ifile[ifile.rfind("/") + 1:len(ifile)]
    ofile = "{0}/preproc_{1}".format(ifile[0:ifile.rfind("/")], ifile_name)
    print "Input file:\t{0}\nOutput file:\t{1}".format(ifile, ofile)
    preprocess(ifile, ofile)

