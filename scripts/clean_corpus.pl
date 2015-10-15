#!/usr/bin/env perl

use strict;
use warnings;

my $idir = shift @ARGV;
my $odir = shift @ARGV;

my @files = `find $idir -type f -name "*.txt"`;
chomp @files;

foreach my $ifile (@files) {
  print STDERR "Parsing $ifile\n";

  $ifile =~ m/.+\/([^\/]+)\.txt$/;
  my $ofile = $odir . "/" . $1 . ".txt";

  my $rc = system "python clean_corpus.py $ifile $ofile";
  die "Error running clean_corpus.py on $ifile: $!" if ($rc >> 8) != 0;
}
