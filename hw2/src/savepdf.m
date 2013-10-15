function savepdf( filename, width, height, h)
%savepdf( filename, width, height, h) resize figure and save to cropped pdf
%   filename    string with name of file to save as
%   width       int indicating number of inches for width.  defaults to 6
%   height      int indicating number of inches for height.  defaults to 4
%   h           figure handle.  defaults to gcf.

if ~exist('width', 'var'); width = 6; end;
if ~exist('height', 'var'); height = 4; end;
if ~exist('h', 'var'); h = gcf; end;

set(gcf, 'Units', 'inches', 'Position', [ 0, 0, width, height] )
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperSize', [width height]);
set(gcf, 'PaperPositionMode', 'auto');
saveas(h, filename, 'pdf')

end