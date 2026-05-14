// This Rust source file is a multiple threaded implementation to perform an
// extremely fast Segmented Sieve of Zakiya (SSoZ) to find Twin Primes <= N.

// Inputs are single values N, or ranges N1 and N2, of 64-bits, 0 -- 2^64 - 1.
// Output is the number of twin primes <= N, or in range N1 to N2; the last
// twin prime value for the range; and the total time of execution.

// Can compile as: $ cargo build --release
// or: $ RUSTFLAGS="-C opt-level=3 -C debuginfo=0 -C target-cpu=native" cargo build --release
// The later compilation creates faster runtime on my system.
// To reduce binary size in target/release/ do: $ strip twinprimes_ssoz
// Single val: $ ./twinprimes_ssoz  val
// Range vals: $ ./twinprimes_ssoz  val1  val2
// val1 and val2 can now be entered as either: 123456789 or 123_456_789

// Mathematical and technical basis for implementation are explained here:
// https://www.academia.edu/37952623/The_Use_of_Prime_Generators_to_Implement_Fast_
// Twin_Primes_Sieve_of_Zakiya_SoZ_Applications_to_Number_Theory_and_Implications_
// for_the_Riemann_Hypotheses
// https://www.academia.edu/7583194/The_Segmented_Sieve_of_Zakiya_SSoZ_
// https://www.academia.edu/19786419/PRIMES-UTILS_HANDBOOK
// https://www.academia.edu/81206391/Twin_Primes_Segmented_Sieve_of_Zakiya_SSoZ_Explained

// This source code, and updates, can be found here:
// https://gist.github.com/jzakiya/b96b0b70cf377dfd8feb3f35eb437225

// Significant contributions provided from https://users.rust-lang.org/
// This code is provided free and subject to copyright and terms of the
// GNU General Public License Version 3, GPLv3, or greater.
// License copy/terms are here: http://www.gnu.org/licenses/

// Copyright (c) 2017-2026; Jabari Zakiya -- jzakiya at gmail dot com
// Last update: 2026/01/20

extern crate integer_sqrt;
extern crate num_cpus;
extern crate rayon;
use integer_sqrt::IntegerSquareRoot;
use rayon::prelude::*;
use std::sync::atomic::{self, AtomicUsize, AtomicU8, Ordering};
use std::time::SystemTime;

// A counter implemented using relaxed (unsynchronized) atomic operations.
struct RelaxedCounter(AtomicUsize);
impl RelaxedCounter {
  fn new() -> Self { RelaxedCounter(AtomicUsize::new(0)) }
  /// Increment and get the new value.
  fn increment(&self) -> usize {
    self.0.fetch_add(1, atomic::Ordering::Relaxed) + 1 }
}

fn print_time(title: &str, time: SystemTime) {
  print!("{} = ", title);
  println!("{} secs", {
    match time.elapsed() {
      Ok(e) => e.as_secs() as f64 + e.subsec_nanos() as f64 / 1_000_000_000f64,
      Err(e) => { panic!("Timer error {:?}", e) },
    }
  });
}

// Customized gcd to determine coprimality; n > m; m odd
fn coprime(mut m: usize, mut n: usize) -> bool {
  while m|1 != 1 { let t = m; m = n % m; n = t }
  m > 0
}

// Compute modular inverse a^-1 to base m, e.g. a*(a^-1) mod m = 1
fn modinv(a0: usize, m0: usize) -> usize {
  if m0 == 1 { return 1 }
  let (mut a, mut m) = (a0 as isize, m0 as isize);
  let (mut x0, mut inv) = (0, 1);
  while a > 1 {
    inv -= (a / m) * x0;
    a = a % m;
    std::mem::swap(&mut a, &mut m);
    std::mem::swap(&mut x0, &mut inv);
  }
  if inv < 0 { inv += m0 as isize }
  inv as usize
}

fn gen_pg_parameters(prime: usize) -> (usize, usize, usize, Vec<usize>, Vec<usize>) {
  // Create prime generator parameters for given Pn
  println!("using Prime Generator parameters for P{}", prime);
  let primes: Vec<usize> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23];
  let (mut modpg, mut res_0) = (1, 0);      // compute Pn's modulus and res_0 value
  for prm in primes { res_0 = prm; if prm > prime { break };  modpg *= prm }

  let mut restwins: Vec<usize> = vec![];    // save upper twinpair residues here
  let mut inverses = vec![0usize; modpg+2]; // save Pn's residues inverses here
  let (mut rc, mut inc, mut res) = (5,2,0); // use P3's PGS to generate residue candidates
  while rc < (modpg >> 1) {                 // find PG's 1st half residues
    if coprime(rc, modpg) {                 // if residue candidate is a residue
      let mc = modpg - rc;                  // create its modular complement
      inverses[rc] = modinv(rc, modpg);     // save rc and mc inverses
      inverses[mc] = modinv(mc, modpg);     // if in twinpair save both hi residues
      if res + 2 == rc { restwins.push(rc); restwins.push(mc + 2) }
      res = rc;                             // save current found residue
    }
    rc += inc; inc ^= 0b110;                // create next P3 sequence rc: 5 7 11 13 17 19 ...
  }
  restwins.sort();         restwins.push(modpg + 1);        // last residue is last hi_tp
  inverses[modpg + 1] = 1; inverses[modpg - 1] = modpg - 1; // last 2 residues are self inverses
  (modpg, res_0, restwins.len(), restwins, inverses)
}

fn set_sieve_parameters(start_num: usize, end_num: usize) ->
  (usize, usize, usize, usize, usize, usize, usize, Vec<usize>, Vec<usize>) {
  // Select at runtime best PG and segment size parameters for input values.
  // These are good estimates derived from PG data profiling. Can be improved.
  let nrange = end_num - start_num;
  let bn: usize; let pg: usize;
  if end_num < 49 {
    bn = 1; pg = 3;
  } else if nrange < 77_000_000 {
    bn = 16; pg = 5;
  } else if nrange <  1_100_000_000 {
    bn = 32; pg = 7;
  } else if nrange < 35_500_000_000 {
    bn = 64; pg = 11;
  } else if nrange < 14_000_000_000_000 {
    pg = 13;
    if      nrange > 7_000_000_000_000 { bn = 384; }
    else if nrange > 2_500_000_000_000 { bn = 320; }
    else if nrange >   250_000_000_000 { bn = 196; }
    else { bn = 128; }
  } else {
    bn = 384; pg = 17;
  }
  let (modpg, res_0, pairscnt, restwins, resinvrs) = gen_pg_parameters(pg);
  let kmin = (start_num-2) / modpg + 1;  // number of resgroups to start_num
  let kmax = (end_num - 2) / modpg + 1;  // number of resgroups to end_num
  let krange = kmax - kmin + 1;          // number of resgroups in range, at least 1
  let n = if krange < 37_500_000_000_000 { 10 } else if krange < 975_000_000_000_000 { 16 } else { 20 };
  let b = bn * 1024 * n;                 // set seg size to optimize for selected PG
  let ks = if krange < b { krange } else { b }; // segments resgroups size

  println!("segment size = {} resgroups; seg array is [1 x {}] 64-bits", ks, ((ks-1) >> 6) + 1);
  let maxpairs = krange * pairscnt;      // maximum number of twinprime pcs
  println!("twinprime candidates = {}; resgroups = {}", maxpairs, krange);
  (modpg, res_0, ks, kmin, kmax, krange, pairscnt, restwins, resinvrs)
}

fn atomic_slice(slice: &mut [u8]) -> &[AtomicU8] {
    unsafe { &*(slice as *mut [u8] as *const [AtomicU8]) }
}

fn sozp5(val: usize, res_0: usize, start_num : usize, end_num : usize) -> Vec<usize> {
  // Return the primes r0..sqrt(end_num) within range (start_num...end_num)
  let (md, rescnt) = (30, 8);            // P5's modulus and residues count
  static RES: [usize; 8] = [7,11,13,17,19,23,29,31];
  static BITN: [u8; 30] = [0,0,0,0,0,1,0,0,0,2,0,4,0,0,0,8,0,16,0,0,0,32,0,0,0,0,0,64,0,128];
  let range_size = end_num - start_num;  // integers size of inputs range

  let kmax = (val - 2) / md + 1;         // number of resgroups upto input value
  let mut prms = vec![0u8; kmax];        // byte array of prime candidates, init '0'
  let sqrtn = val.integer_sqrt();        // compute integer sqrt of val
  let k = sqrtn / md;                    // compute its resgroup value
  let (resk, mut r) = (sqrtn - md*k, 0); // compute its residue value; set residue start posn
  while resk >= RES[r] { r += 1 }        // find largest residue <= sqrtn posn in its resgroup
  let pcs_to_sqrtn = k*rescnt + r;       // number of pcs <= sqrtn

  for i in 0..pcs_to_sqrtn {             // for r0..sqrtN primes mark their multiples
    let (k, r) = (i/rescnt,  i%rescnt);  // update resgroup parameters
    if (prms[k] & (1 << r)) != 0 { continue } // skip pc if not prime
    let prm_r = RES[r];                  // if prime save its residue value
    let prime = md*k + prm_r;            // numerate its value
    let rem = start_num % prime;         // prime's modular distance to start_num
    if  !(prime - rem <= range_size || rem == 0) { continue } // skip prime if no multiple in range
    let prms_atomic = atomic_slice(&mut prms); // to parallel sieve along bit rows
    RES.par_iter().for_each (|ri| {      // mark prime's multiples in prms in parallel
      let prod = prm_r * ri - 2;         // compute cross-product for prm_r|ri pair
      let bit_r = BITN[prod % md];       // bit mask value for prod's residue
      let mut kpm = k * (prime + ri) + prod / md; // resgroup for prime's 1st multiple
      while kpm < kmax { prms_atomic[kpm].fetch_or(bit_r, Ordering::Relaxed); kpm += prime; };
    });
  }
  // prms now contains the prime multiples positions marked for the pcs r0..N
  // in parallel along each restrack, identify|numerate the primes in each resgroup
  // return only the primes with a multiple within range (start_num...end_num)
  let primes = RES.par_iter().enumerate().flat_map_iter( |(i, ri)| {
    prms.iter().enumerate().filter_map(move |(k, resgroup)| {
      if resgroup & (1 << i) == 0 {
        let prime = md * k + ri;
        let rem = start_num % prime;
        if (prime >= res_0 && prime <= val) && (prime - rem <= range_size || rem == 0) {
          return Some(prime);
      } } None
  }) }).collect();
  primes
}

fn nextp_init(rhi: usize, kmin: usize, modpg: usize, primes: &[usize], resinvrs: &[usize]) -> Vec<usize> {
  // Initialize 'nextp' array for twinpair upper residue rhi in 'restwins'.
  // Compute 1st prime multiple resgroups for each prime r0..sqrt(N) and
  // store consecutively as lo_tp|hi_tp pairs for their restracks.
  let mut nextp = vec![0usize; primes.len() * 2]; // 1st mults array for twinpair
  let (r_hi, r_lo) = (rhi, rhi - 2);          // upper|lower twinpair residue values
  for (j, prime) in primes.iter().enumerate() { // for each prime r0..sqrt(N)
    let k = (prime - 2) / modpg;              // find the resgroup it's in
    let r = (prime - 2) % modpg + 2;          // and its residue value
    let r_inv = resinvrs[r];                  // and residue inverse
    let rl = (r_lo * r_inv - 2) % modpg + 2;  // compute r's rl for r_lo
    let rh = (r_hi * r_inv - 2) % modpg + 2;  // compute r's rh for r_hi
    let mut kl = k * (prime + rl) + (r * rl - 2) / modpg; // kl 1st mult resgroup
    let mut kh = k * (prime + rh) + (r * rh - 2) / modpg; // kh 1st mult restroup
    if kl < kmin { kl = (kmin - kl) % prime; if kl > 0 { kl = prime - kl } }
    else { kl = kl - kmin };
    if kh < kmin { kh = (kmin - kh) % prime; if kh > 0 { kh = prime - kh } }
    else { kh = kh - kmin };
    nextp[j * 2] = kl;                        // prime's 1st mult lo_tp resgroups in range
    nextp[j * 2 | 1] = kh;                    // prime's 1st mult hi_tp resgroups in range
  }
  nextp
}

fn twins_sieve(r_hi: usize, kmin: usize, kmax: usize, ks: usize, start_num: usize,
  end_num: usize, modpg: usize, primes: &[usize], resinvrs: &[usize]) -> (usize, usize) {
  // Perform in thread the ssoz for given twinpair residues for kmax resgroups.
  // First create|init 'nextp' array of 1st prime mults for given twinpair,
  // stored consequtively in 'nextp', and init seg array for ks resgroups.
  // For sieve, mark resgroup bits to '1' if either twinpair restrack is nonprime
  // for primes mults resgroups, and update 'nextp' restrack slices acccordingly.
  // Return the last twinprime|sum for the range for this twinpair residues.
  // For speed, disable runtime seg array bounds checking; using 64-bit elem seg array
  unsafe {                                                    // allow fast array indexing
    type MWord = u64;                                         // mem size for 64-bit cpus
    const S: usize = 6;                                       // shift value for 64 bits
    const BMASK: usize = (1 << S) - 1;                        // bitmask val for 64 bits
    let (mut sum, mut ki, mut kn) = (0usize, kmin-1, ks);     // init these parameters
    let (mut hi_tp, mut k_max) = (0usize, kmax);              // max twinprime|resgroup val
    let mut seg = vec![0 as MWord; ((ks - 1) >> S) + 1];      // seg array for ks resgroups
    if r_hi - 2 < (start_num - 2) % modpg + 2 { ki += 1; }    // ensure lo tp in range
    if r_hi > (end_num - 2) % modpg + 2 { k_max -= 1; }       // ensure hi tp in range
    let mut nextp = nextp_init(r_hi, ki, modpg, primes, resinvrs); // init nextp array
    while ki < k_max {                                 // for ks size slices upto kmax
      if ks > (k_max - ki) { kn = k_max - ki }         // adjust kn size for last slice
      for (j, prime) in primes.iter().enumerate() {    // for each prime r0..sqrt(N)
                                                       // for lower twinpair residue track
        let mut k = *nextp.get_unchecked(j * 2);       // starting from this resgroup in seg
        while k < kn  {                                // mark primenth resgrouup bits prime mults
          *seg.get_unchecked_mut(k >> S) |= 1 << (k & BMASK);
          k += prime; }                                // set resgroup for prime's next multiple
        *nextp.get_unchecked_mut(j * 2) = k - kn;      // save 1st resgroup in next eligible seg
                                                       // for upper twinpair residue track
        k = *nextp.get_unchecked(j * 2 | 1);           // starting from this resgroup in seg
        while k < kn  {                                // mark primenth resgroup bits prime mults
          *seg.get_unchecked_mut(k >> S) |= 1 << (k & BMASK);
          k += prime; }                                // set resgroup for prime's next multiple
        *nextp.get_unchecked_mut(j * 2 | 1) = k - kn;  // save 1st resgroup in next eligible seg
      }                                                // set as nonprime unused bits in last seg[n]
                                                       // so fast, do for every seg[i]
      *seg.get_unchecked_mut((kn - 1) >> S) |= !1u64 << ((kn - 1) & BMASK);
      let mut cnt = 0usize;                            // count the twinprimes in the segment
      for &m in &seg[0..=(kn - 1) >> S] { cnt += m.count_zeros() as usize; }
      if cnt > 0 {                                     // if segment has twinprimes
        sum += cnt;                                    // add the segment count to total count
        let mut upk = kn - 1;                          // from end of seg count back to largest tp
        while *seg.get_unchecked(upk >> S) & (1 << (upk & BMASK)) != 0 { upk -= 1 }
        hi_tp = ki + upk;                              // set resgroup value for largest tp in seg
      }
      ki += ks;                                        // set 1st resgroup val of next seg slice
      if ki < k_max { seg.fill(0); }                   // set seg to all primes
    }                                                  // when sieve done, numerate largest tp in range
                                                       // for small ranges w/o twins, set largest to 1
    hi_tp = if r_hi > end_num || sum == 0 { 1 } else { hi_tp * modpg + r_hi };
    (hi_tp, sum)                                       // return largest twinprime|twins count
  }
}

fn main() {
  let nums: Vec<String> = std::env::args().collect();
  let n1: usize = nums[1].replace('_', "").parse().unwrap();
  let n2: usize = if nums.len() < 3 { 3 } else { nums[2].replace('_', "").parse().unwrap() };

  let mut end_num = std::cmp::max(n1, 3); // min vals 3
  let mut start_num = std::cmp::max(n2, 3);
  if start_num > end_num { std::mem::swap(&mut end_num, &mut start_num) }
  start_num |= 1;                        // if start_num even increase by 1
  end_num = (end_num - 1) | 1;           // if end_num even decrease by 1
  if end_num - start_num < 2 { end_num = 7; start_num = 7 }

  println!("threads = {}", num_cpus::get());
  let ts = SystemTime::now();            // start timing sieve setup execution
                                         // select Pn, set sieving params for inputs
  let (modpg, res_0, ks, kmin, kmax, krange, pairscnt, restwins, resinvrs) = set_sieve_parameters(start_num, end_num);

  // create sieve primes <= sqrt(end_num), only use those whose multiples within inputs range
  let primes: Vec<usize> = if end_num < 49 { vec![5] }
                           else { sozp5(end_num.integer_sqrt(), res_0, start_num, end_num) };

  println!("each of {} threads has nextp[2 x {}] array", pairscnt, primes.len());

  let mut twinscnt = 0usize;             // init twinprimes range count
  let lo_range = restwins[0] - 3;        // lo_range = lo_tp - 1
  for tp in [3, 5, 11, 17] {             // excluded low tp values PGs used
    if end_num == 3 { break };           // if 3 end of range, no twinprimes
    if tp >= start_num && tp <= lo_range { twinscnt += 1 }; // cnt any small tps
  }

  print_time("setup time", ts);          // sieve setup time
  println!("perform twinprimes ssoz sieve");
  let t1 = SystemTime::now();            // start timing ssoz sieve execution
                                         // sieve each twinpair restracks in parallel
  let (lastwins, cnts): (Vec<_>, Vec<_>) = { // store outputs in these arrays
    let counter = RelaxedCounter::new();
    restwins.par_iter().map( |r_hi| {
      let out = twins_sieve(*r_hi, kmin, kmax, ks, start_num, end_num, modpg, &primes, &resinvrs);
      print!("\r{} of {} twinpairs done", counter.increment(), pairscnt);
      out
    }).unzip()
  };
  let mut last_twin = 0usize;            // find largest twinprime|cnts in range
  for (i, cnt_i) in cnts.iter().enumerate() {
    twinscnt += cnt_i;
    if last_twin < lastwins[i] { last_twin = lastwins[i]; }
  }
  if end_num == 5 && twinscnt == 1 { last_twin = 5; }
  let mut kn = krange % ks;              // set number of resgroups in last slice
  if kn == 0 { kn = ks };                // if multiple of seg size set to seg size

  print_time("\nsieve time", t1);        // ssoz sieve time
  print_time("total time", ts);          // setup + sieve time
  println!("last segment = {} resgroups; segment slices = {}", kn, (krange - 1)/ks + 1);
  println!("total twins = {}; last twin = {}|-2", twinscnt, last_twin);
}
