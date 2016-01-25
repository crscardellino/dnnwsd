#!/usr/bin/env perl

use strict;
use warnings;

my @files = `find tuning_accuracy -type f`;
chomp @files;

my $outdir = "experiments_tuning_accuracy";
my $tmpdir = "/tmp/tuning_accuracy";
system "rm -rf $outdir $tmpdir";
mkdir $outdir;
mkdir $tmpdir;

my $index = 0; 

foreach my $file(@files) {
    system "cut -d',' -f1 $file > $tmpdir/${index}_bow";
    system "cut -d',' -f2 $file > $tmpdir/${index}_wordvec";
    system "cut -d',' -f3 $file > $tmpdir/${index}_wordvecpos";

    $index++;
}

$index--;

system "paste -d',' $tmpdir/{0..$index}_bow | sed 's/^,/NA,/g' | sed 's/,\$/,NA/g' | sed 's/,,/,NA,/g' | sed 's/,,/,NA,/g' > bow.csv";

