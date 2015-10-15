#!/usr/bin/env perl

use strict;
use warnings;

my $idir = shift @ARGV;
my $odir = shift @ARGV;

my @files = `find $idir -type f -name "*.xml"`;
chomp @files;

foreach my $ifile (@files) {
  print STDERR "Parsing $ifile\n";

  $ifile =~ m/.+\/([^\/]+)\.xml$/;
  my $ofile = $odir . "/" . $1 . ".txt";

  my $rc = system "python jrc.py $ifile $ofile";
  die "Error running jrc.py on $ifile: $!" if ($rc >> 8) != 0;
}
