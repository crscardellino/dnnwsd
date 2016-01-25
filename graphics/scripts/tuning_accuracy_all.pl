#!/usr/bin/env perl

use strict;
use warnings;

my @files = `find tuning_accuracy -type f`;
chomp @files;

my $verb_file = "../../resources/sensem/verbs";
my $outdir = "../tuning_accuracy_plots";

system "rm -rf $outdir";
mkdir $outdir;

foreach my $file(@files) {
    $file =~ m/tuning_accuracy\/([0-9]+)/;
    my $verb_idx = $1;

    my $verb_cmd = sprintf("sed -n '%d,%dp' %s", $verb_idx+1, $verb_idx+1, $verb_file);
    my $verb = `$verb_cmd`; 
    chomp $verb;

    print STDERR "Plotting data for lemma $verb\n";

    system "./tuning_accuracy.R $verb $file $outdir/$verb_idx.pdf &> /dev/null";
}
