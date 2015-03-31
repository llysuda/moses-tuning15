#!/usr/bin/perl 

if ($#ARGV < 1) {
    die "\$ nbest-for-sa ref extend > nbest.out 2> ref.out ";
}

my $nbest = $ARGV[0];
my $ref = $ARGV[1];
my $extend = $ARGV[2];

# read references
my @refLines;
open R, "<", $ref ;
while (<R>) {
    chomp;
    push @refLines, $_;
}
close R;

# read nbest, reformat, output
#
if ($nbest =~ /\.gz/) {
    open N, "gunzip -c $nbest | ";
} else {
    open N, "<", $nbest ;
}

my $count = 0;
my $prevSpan = "0 0";
my $prevSentId = 0;
my $lcount = 0;
while (<N>) {
    chomp;
    my @items = split /\s+\|\|\|\s+/;
    my $sentId = $items[0];
    my $target = $items[1];
    my $features = $items[2];
    my $score = $items[3];
    my $span = $items[4];
    
    my $potTrans = "";
    if ($extend) {
        $potTrans = $items[5];
    }
    
    # remove possible space
    $span =~ s/^\s+//g;
    $span =~ s/\s+$//g;
    $span =~ s/\s+/ /g;
    
    # new sentence
    if ($prevSentId ne $sentId || $prevSpan ne $span) {
        if ($lcount > 0) {
            $count++;
        }
        $prevSentId = $sentId;
        $prevSpan = $span;
        print STDERR "$refLines[$sentId]\n";
    }
    
    if ($extend) {
        print STDOUT "$count ||| $potTrans ||| $features ||| $score\n";
    } else {
        print STDOUT "$count ||| $target ||| $features ||| $score\n";
    }
    
    $lcount++;
}
