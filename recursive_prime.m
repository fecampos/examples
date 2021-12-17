function [out] = prime_number(n)

out = 'true'; s = length(n);

if s == 1;
  n = [n round(n/2)+1];
end

a = n(1); b = n(2);

if b < 2
  out = 'true'; return
elseif a == 2 || a == 3
  out = 'true'; return;
elseif mod(a,b) == 0
  out = 'false'; return;
else
  b = b-1; 
end

out = prime_number([a b]);

return
