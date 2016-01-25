#!/usr/bin/env perl

use strict;
use warnings;

my $directory = "../resources/results/results_semisupervised";
my $outdir = "tuning_accuracy";
print STDERR "Getting data from $directory and saving it to $outdir\n";

my @verbs = `cat ../resources/sensem/verbs`;
chomp @verbs;

my $index = 0; 

foreach my $verb(@verbs) {
    my $verb_dir = sprintf("%s/%03d", $directory, $index);

    if (not -d $verb_dir) {
        print STDERR "Skipping $verb_dir of verb $verb. Non-existing directory.\n";
        $index++;
        next;
    }

    open(my $fh, ">", sprintf("$outdir/%03d", $index));
    print $fh "bow,wordvec,wordvecpos\n";
    
    my @experiments = map { "$verb_dir/$_/accuracy" } qw/bow_logreg wordvec_mlp_2_0 wordvecpos_mlp_2_0/;

    my $cmd = "paste -d',' " . join(" ", @experiments) . " | sed 's/^,/NA,/g' | sed 's/,\$/,NA/g' | sed 's/,,/,NA,/g'";

    my $data = `$cmd`;
    
    print $fh $data;

    $index++;
    close $fh;
}

