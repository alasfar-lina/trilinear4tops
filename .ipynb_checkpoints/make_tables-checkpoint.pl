use LaTeX::Table;
use Number::Format qw(:subs);  # use mighty CPAN to format values
use YAML qw(LoadFile);


# Open the config
$yaml =  LoadFile( '../data.yaml' );
my $header = [
    [ '$\delta R(C_i)\cdot 10^{-2}$', '$C_{Qt}^{(1)}$','$C_{Qt}^{(8)}$', '$C_{QtQb}^{(1)}$','$C_{QtQb}^{(8)}$','$C_{tt}^{(1)}$','$C_{QQ}^{(1)}$','$C_{QQ}^{(8)}$' ],
];

my $data = [
    [ 'ggF/ $gg\to h$',      $yaml->{Cqt1}->{ggFos},  $yaml->{Cqt8}->{ggFos} ,$yaml->{Cqtqb1}->{ggFos} ,$yaml->{Cqtqb8}->{ggFos},'-','-','-' ],
    #------------------------
    [ '$t\bar{t}h$', $yaml->{Cqt1}->{ttH},  $yaml->{Cqt8}->{ttH} ,$yaml->{Cqtqb1}->{ttH} ,$yaml->{Cqtqb8}->{ttH},$yaml->{Ctt1}->{ttH},$yaml->{Cqq1}->{ttH},$yaml->{Cqq8}->{ttH} ],
    #------------------------
    [ '$h\to \gamma \gamma$',  $yaml->{Cqt1}->{gagaos},  $yaml->{Cqt8}->{gagaos} ,$yaml->{Cqtqb1}->{gagaos} ,$yaml->{Cqtqb8}->{gagaos},'-','-','-' ],
    #------------------------
    [ '$h\to b\bar{b}$',  '-',  '-' ,$yaml->{Cqtqb1}->{Hbb} ,$yaml->{Cqtqb8}->{Hbb},'-','-','-' ],
    #------------------------
];

my $table = LaTeX::Table->new(
      {
      filename    => './results/tables/results4top.tex',
      maincaption => '',
      caption     => 'The relative correction dependence on the $C_i$ for Higgs processes ',
      label       => 'table:res4top',
      position    => 'tbp!',
      header      => $header,
      data        => $data,
      }
);

$table->set_callback(sub {
     my ($row, $col, $value, $is_header ) = @_;
     if ($col == 0 || $is_header) {
         $value =  $value;
     }
     elsif ($col > 0 && !$is_header) {
         $value = format_number($value*100, 3,1);
     }
     return $value;
});

# write LaTeX code in tex file
$table->generate();

print $table->generate_string();
#===================================================
my $header2 = [
    [ '$\delta R(C_i)\cdot 10^{-2}$', '$C_\phi$' ],
];

my $data2 = [
    [ 'ggF/ $gg\to h$',$yaml->{CH}->{ggF} ],
    #------------------------
    [ '$t\bar{t}h$',$yaml->{CH}->{ttH} ],
    #------------------------
    [ '$gg\to \gamma$',$yaml->{CH}->{gaga} ],
    #------------------------
    [ '$gg\to b\bar{b}$',$yaml->{CH}->{ff} ],
    #------------------------
    [ '$gg\to W^+ W^-$',$yaml->{CH}->{ww} ],
    #------------------------
    [ '$gg\to Z Z$',$yaml->{CH}->{zz} ],
    #------------------------
    [ '$pp\to Zh$',$yaml->{CH}->{ZH} ],
    #------------------------
    [ '$pp\to W^\pm h$',$yaml->{CH}->{WH} ],
    #------------------------
    [ 'VBF',$yaml->{CH}->{VBF} ],
];

my $table2 = LaTeX::Table->new(
      {
      filename    => './results/tables/resultsCH.tex',
      maincaption => '',
      caption     => 'The relative correction dependence on the $C_H$ for Higgs processes taken from~\cite{ Degrassi:2021uik} ',
      label       => 'table:resch',
      position    => 'tbp!',
      header      => $header2,
      data        => $data2,
      }
);

$table2->set_callback(sub {
     my ($row, $col, $value, $is_header ) = @_;
     if ($col == 0 || $is_header) {
         $value =  $value;
     }
     elsif ($col > 0 && !$is_header) {
         $value = format_number($value*100, 2,1);
     }
     return $value;
});

# write LaTeX code in tex file
$table2->generate();

print $table2->generate_string();

#===================================================
my $header3 = [
    [ ' ' ,'Observable', 'value', 'uncertainty' ],
];


my $data3 = [
#-------------------------------------------------- ggF
    [ '\multirow{4}{*}{ ATLAS ggF}','$ \mu (h\to \gamma  \gamma)$'$yaml->{Bounds}->{ATLAS}->{ggf}->{mu_gaga}, $yaml->{Bounds}->{ATLAS}->{ggf}->{err_gaga} ],
    [ '','$ \mu (h\to Z Z)$',$yaml->{Bounds}->{ATLAS}->{ggf}->{mu_zz}, $yaml->{Bounds}->{ATLAS}->{ggf}->{err_zz} ],
    [ '','$ \mu (h\to W^+W^-)$',$yaml->{Bounds}->{ATLAS}->{ggf}->{mu_ww}, $yaml->{Bounds}->{ATLAS}->{ggf}->{err_ww} ],
    [ '','$ \mu (h\to \tau^+\tau^- )$',$yaml->{Bounds}->{ATLAS}->{ggf}->{mu_tata}, $yaml->{Bounds}->{ATLAS}->{ggf}->{err_tata}],
	[], # midrule
#-------------------------------------------------- VBF
    [ '\multirow{5}{*}{ ATLAS VBF}','$ \mu (h\to \gamma  \gamma)$',
    $yaml->{Bounds}->{ATLAS}->{vbf}->{mu_gaga}, $yaml->{Bounds}->{ATLAS}->{vbf}->{err_gaga} ],
	[ '','$ \mu (h\to Z Z)$',$yaml->{Bounds}->{ATLAS}->{vbf}->{mu_zz}, $yaml->{Bounds}->{ATLAS}->{vbf}->{err_zz} ],
	[ '','$ \mu (h\to W^+W^-)$',$yaml->{Bounds}->{ATLAS}->{vbf}->{mu_ww}, $yaml->{Bounds}->{ATLAS}->{vbf}->{err_ww} ],
	[ '','$ \mu (h\to \tau^+\tau^- )$',$yaml->{Bounds}->{ATLAS}->{vbf}->{mu_tata}, $yaml->{Bounds}->{ATLAS}->{vbf}->{err_tata}]
	[ '','$ \mu (h\to  b \bar b)$',$yaml->{Bounds}->{ATLAS}->{vbf}->{mu_bb}, $yaml->{Bounds}->{ATLAS}->{vbf}->{err_bb}]
	[],
#-------------------------------------------------- ttH
    [ '\multirow{4}{*}{ ATLAS $t\bar t h$}','$ \mu (h\to \gamma  \gamma)$',
    $yaml->{Bounds}->{ATLAS}->{ttxh}->{mu_gaga}, $yaml->{Bounds}->{ATLAS}->{vbf}->{err_gaga} ],
	[ '','$ \mu (h\to Z Z)$',$yaml->{Bounds}->{ATLAS}->{ttxh}->{mu_vv}, $yaml->{Bounds}->{ATLAS}->{ttxh}->{err_vv} ],
	[ '','$ \mu (h\to \tau^+\tau^- )$',$yaml->{Bounds}->{ATLAS}->{ttxh}->{mu_tata}, $yaml->{Bounds}->{ATLAS}->{ttxh}->{err_tata}]
	[ '','$ \mu (h\to  b \bar b)$',$yaml->{Bounds}->{ATLAS}->{ttxh}->{mu_bb}, $yaml->{Bounds}->{ATLAS}->{ttxh}->{err_bb}]
	[],
#-------------------------------------------------- VH
	[ '\multirow{3}{*}{ ATLAS $Vh$}','$ \mu (h\to \gamma  \gamma)$',  $yaml->{Bounds}->{ATLAS}->{vh}->{mu_gaga}, $yaml->{Bounds}->{ATLAS}->{vh}->{err_gaga} ],
	[ '','$ \mu (h\to Z Z)$',$yaml->{Bounds}->{ATLAS}->{vh}->{mu_zz}, $yaml->{Bounds}->{ATLAS}->{vh}->{err_zz} ],
	[ '','$ \mu (h\to  b \bar b)$',$yaml->{Bounds}->{ATLAS}->{vh}->{mu_bb}, $yaml->{Bounds}->{ATLAS}->{vh}->{err_bb}]
	[],
    ### ADD CMS and correct 4 top
#-------------------------------------------------- 4 top
#[ '\multirow{4}{*}{Global fit}','$\cqu^{(1)} $',$yaml->{Bounds}->{cqt1}, $yaml->{Bounds}->{cqt1_delta} ],
#[ '','$\cqu^{(8)}$',$yaml->{Bounds}->{cqt8}, $yaml->{Bounds}->{cqt8_delta} ],
#[ '','$\cquqd^{(1)}$',$yaml->{Bounds}->{cqtqb1}, $yaml->{Bounds}->{cqtqb1_delta} ],
#[ '','$\cquqd^{(8)}$',$yaml->{Bounds}->{cqtqb8}, $yaml->{Bounds}->{cqtqb8_delta} ],
];



my $table3 = LaTeX::Table->new(
      {
      filename    => './results/tables/fitobservables.tex',
      maincaption => '',
      caption     => 'The observable values and their uncertainties used in the fit ',
      label       => 'table:resch',
      position    => 'tbp!',
      header      => $header3,
      data        => $data3,
      }
);

$table3->set_callback(sub {
     my ($row, $col, $value, $is_header ) = @_;
     if ($col == 0 || $is_header) {
         $value =  $value;
     }
     elsif ($col > 1 && !$is_header) {
         $value = format_number($value, 3,1);
     }
     return $value;
});

# write LaTeX code in tex file
$table3->generate();

print $table3->generate_string();
