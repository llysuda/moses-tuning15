#!/usr/bin/perl 

my %ranges;
while (<STDIN>) {
    chomp;
    my @items = split /\s+\|\|\|\s+/;
    my $sentid = $items[0];
    my @points = split /\s+/, $items[$#items];
    for(my $i=0; $i < $#points; ) {
        my $key = "$sentid $points[$i] $points[$i+1]";
        if (!defined $ranges{$key}) {
            print "$key\n";
            $ranges{$key} = 1;
        }
        $i += 2;
    }
}
