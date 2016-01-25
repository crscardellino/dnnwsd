#!/usr/bin/env perl

use strict;
use warnings;

my $directory = "../resources/results/results_semisupervised";
my $file = "test_accuracy.csv";
print STDERR "Getting data from $directory and saving it to $file\n";

my @verbs = `cat ../resources/sensem/verbs`;
chomp @verbs;

my $index = 0; 

open(my $fh, ">", $file);

print $fh "verb,bow_initial,bow_final,wordvec_initial,wordvec_final,wordvecpos_initial,wordvecpos_final\n";

foreach my $verb(@verbs) {
    my $verb_dir = sprintf("%s/%03d", $directory, $index);

    if (not -d $verb_dir) {
        print STDERR "Skipping $verb_dir of verb $verb. Non-existing directory.\n";
        $index++;
        next;
    }

    print $fh "$verb";
    
    foreach my $experiment(qw/bow_logreg wordvec_mlp_2_0 wordvecpos_mlp_2_0/){
        my @accuracy = `cat $verb_dir/$experiment/test_accuracy`;
        chomp @accuracy;

        print $fh ",$accuracy[0]";
        print $fh ",$accuracy[1]";
    }

    print $fh "\n";

    $index++;
}

close $fh;
