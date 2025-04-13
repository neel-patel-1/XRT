static inline uLong get_compress_bound(int payload_size){
  z_stream stream;
  int ret = 0;
  uLong maxcompsize;

  memset(&stream, 0, sizeof(z_stream));
  ret = deflateInit2(&stream, Z_BEST_COMPRESSION, Z_DEFLATED, -12, 9, Z_DEFAULT_STRATEGY);
  if (ret != Z_OK) {
    LOG_PRINT( LOG_ERR, "Error deflateInit2 status %d\n", ret);
    return 0;
  }

  maxcompsize = deflateBound(&stream, payload_size);
  deflateEnd(&stream);
  return maxcompsize;
}

static inline void dump_deflate_state(z_stream *stream)
{
  LOG_PRINT(
      LOG_VERBOSE, "deflate: avail_in=%d, total_in=%ld, avail_out=%d, total_out=%ld, next_out=0x%p\n",
         stream->avail_in,
         stream->total_in,
         stream->avail_out,
         stream->total_out,
         stream->next_out);
}

static inline void dump_inflate_state(z_stream *stream)
{
  LOG_PRINT(
      LOG_VERBOSE, "inflate: avail_in=%d, total_in=%ld, avail_out=%d, total_out=%ld, next_out=0x%p\n",
         stream->avail_in,
         stream->total_in,
         stream->avail_out,
         stream->total_out,
         stream->next_out);
}



static inline int gpcore_do_compress(void *dst, void *src, int src_len, int *out_len)
{
  int ret = 0;
  z_stream stream;
  int avail_out = *out_len;

  memset(&stream, 0, sizeof(z_stream));

  /* allocate deflate state */
  ret = deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -12, 9, Z_DEFAULT_STRATEGY);
  if (ret != Z_OK) {
    LOG_PRINT( LOG_ERR, "Error deflateInit2 status %d\n", ret);
    return ret;
  }

  stream.avail_in = src_len;
  stream.next_in = (Bytef *)src;
  stream.avail_out = avail_out;
  stream.next_out = (Bytef *)dst;

  dump_deflate_state(&stream);

  do {
    ret = deflate(&stream, Z_NO_FLUSH);
    dump_deflate_state(&stream);
  }
  while(stream.avail_in > 0 && ret == Z_OK);

  if(ret != Z_STREAM_END) { /* we need to flush, out input was small */
    ret = deflate(&stream, Z_FINISH);
  }
  dump_deflate_state(&stream);


  ret = deflateEnd(&stream);
  if (ret) {
    LOG_PRINT( LOG_ERR, "Error deflateEnd status %d\n", ret);
    return ret;
  }

  *out_len = stream.total_out;
  return ret;
}

static inline int gpcore_do_decompress(void *dst, void *src, uInt src_len, uLong *out_len)
{
	int ret = 0;
	z_stream stream;
  int avail_out  = *out_len;

	memset(&stream, 0, sizeof(z_stream));

	/* allocate inflate state */
	ret = inflateInit2(&stream, -MAX_WBITS);
	if (ret) {
		LOG_PRINT( LOG_ERR, "Error inflateInit2 status %d\n", ret);
		return ret;
	}

	stream.avail_in = src_len;
	stream.next_in = (Bytef *)src;
	stream.avail_out = avail_out;
	stream.next_out = (Bytef *)dst;

  dump_inflate_state(&stream);

	do {
		ret = inflate(&stream, Z_NO_FLUSH);

		if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
			inflateEnd(&stream);
			LOG_PRINT( LOG_ERR, "Error inflate status %d\n", ret);
			return ret;
		}
    dump_inflate_state(&stream);
	} while (ret != Z_STREAM_END);

	ret = inflateEnd(&stream);
	if (ret) {
		LOG_PRINT( LOG_ERR, "Error inflateEnd status %d\n", ret);
		return ret;
	}
  dump_inflate_state(&stream);

	*out_len = stream.total_out;
	return ret;
}

static inline void gpcore_do_deflate_decompress(void *src, void *dst, int src_len, int *out_len)
{
	int ret = 0;
	z_stream stream;
  int avail_out  = *out_len;

	memset(&stream, 0, sizeof(z_stream));

	/* allocate inflate state */
	ret = inflateInit2(&stream, -MAX_WBITS);
	if (ret) {
		LOG_PRINT( LOG_ERR, "Error inflateInit2 status %d\n", ret);
		return;;
	}

	stream.avail_in = (uInt)src_len;
	stream.next_in = (Bytef *)src;
	stream.avail_out = (uInt)avail_out;
	stream.next_out = (Bytef *)dst;

  dump_inflate_state(&stream);

	do {
		ret = inflate(&stream, Z_NO_FLUSH);

		if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
			inflateEnd(&stream);
			LOG_PRINT( LOG_ERR, "Error inflate status %d\n", ret);
			return;;
		}
    dump_inflate_state(&stream);
	} while (ret != Z_STREAM_END);

	ret = inflateEnd(&stream);
	if (ret) {
		LOG_PRINT( LOG_ERR, "Error inflateEnd status %d\n", ret);
		return;;
	}
  dump_inflate_state(&stream);

	*out_len = (int)(stream.total_out);
	return;;
}