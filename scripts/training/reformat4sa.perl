#!/usr/bin/perl 

if ($#ARGV < 1) {
    die "\$ nbest-for-sa ref extend outprefix > ref.out ";
}

my $nbest = $ARGV[0];
my $ref = $ARGV[1];
my $extend = $ARGV[2];
my $outPrefix = $ARGV[3];

my $sa_weight = $ARGV[4];

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

open PAR, ">", "$outPrefix.par";
#open POT, ">", "$outPrefix.pot";

while (<N>) {
    chomp;
    my @items = split /\s+\|\|\|\s+/;
    my $sentId = $items[0];
    my $target = $items[1];
    my $features = $items[2];
    my $score = $items[3];
    my $span = $items[4];
    
    #my $potTrans = "";
    #if ($extend) {
    #    $potTrans = $items[5];
    #}
    
    # remove possible space
    $span =~ s/^\s+//g;
    $span =~ s/\s+$//g;
	my @lengthes = split /\s+/, $span;
	die "error span: $span\n" if $#lengthes < 1;
	$span =~ s/\s+/ /g;
    
    # new sentence
    if ($prevSentId ne $sentId || $prevSpan ne $span) {
        if ($lcount > 0) {
            $count++;
        }
        $prevSentId = $sentId;
        $prevSpan = $span;
        print STDOUT "$refLines[$sentId]\n";
		if ($sa_weight) {
			my $w = $lengthes[1]/$lengthes[0];
			print STDERR "$w\n";	
		}
    }
    
    #if ($extend) {
    #    print POT "$count ||| $potTrans ||| $features ||| $score\n";
    #}
    print PAR "$count ||| $target ||| $features ||| $score\n";
    
    $lcount++;
}

close PAR;
#close POT;
