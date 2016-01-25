#!/usr/bin/env perl

use strict;
use warnings;

my $input_dir = "../../resources/results/results_semisupervised";
my $quoted_input_dir = quotemeta($input_dir);

my @files = `find $input_dir -type f -name "targets_distribution"`;
chomp @files;

my $verb_file = "../../resources/sensem/verbs";
my $outdir = "../population_progress_plots";

my %experiments_names = (
    "bow_logreg" => "Bag of Words",
    "wordvec_mlp_2_0" => "Word Vectors",
    "wordvecpos_mlp_2_0" => "Word Vectors with PoS"
);

system "rm -rf $outdir";
mkdir $outdir;

foreach my $file(@files) {
    $file =~ m/$quoted_input_dir\/([0-9]+)\/([a-z0-9_]+)\/targets_distribution/;
    my $verb_idx = $1;
    my $experiment = $2;

    my $verb_cmd = sprintf("sed -n '%d,%dp' %s", $verb_idx+1, $verb_idx+1, $verb_file);
    my $verb = `$verb_cmd`; 
    chomp $verb;

    my $number_of_iterations = `cut -d',' -f1 $file | uniq | wc -l`;
    chomp $number_of_iterations;

    if($number_of_iterations == 1) {
        print STDERR "Skipping plot for lemma $verb and experiment $experiment\n";
        next;
    }

    print STDERR "Plotting data for lemma $verb and experiment $experiment\n";

    my $cmd = sprintf("./population_progress.R %s \"%s\" %s %s &> /dev/null",
        $verb, $experiments_names{$experiment}, $file, "$outdir/${verb_idx}_$experiment.pdf");

    system $cmd;
}
