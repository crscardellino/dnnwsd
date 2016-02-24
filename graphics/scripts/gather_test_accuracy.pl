#!/usr/bin/env perl

use strict;
use warnings;

my $directory = "../../resources/results/results_semisupervised_semeval_7k_only_verbs";
my $file = "test_accuracy_semeval_verbs_only.csv";
print STDERR "Getting data from $directory and saving it to $file\n";

my @lemmas = `cat ../../resources/semeval/lexelts/lemmas`;
chomp @lemmas;

my $index = 0; 

open(my $fh, ">", $file);

print $fh "lemma,bow_initial,bow_final,wordvec_initial,wordvec_final,wordvecpos_initial,wordvecpos_final\n";

foreach my $lemma(@lemmas) {
    my $lemma_dir = sprintf("%s/%03d", $directory, $index);

    if (not -d $lemma_dir) {
        print STDERR "Skipping $lemma_dir of lemma $lemma. Non-existing directory.\n";
        $index++;
        next;
    }

    print $fh "$lemma";
    
    foreach my $experiment(qw/bow_logreg wordvec_mlp_2_0 wordvecpos_mlp_2_0/){
        my @accuracy = `cat $lemma_dir/$experiment/test_accuracy`;
        chomp @accuracy;

        print $fh ",$accuracy[0]";
        print $fh ",$accuracy[1]";
    }

    print $fh "\n";

    $index++;
}

close $fh;
