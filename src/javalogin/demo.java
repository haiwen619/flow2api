public void reAuth(CookiesEntity account) throws IOException {
        log.info("重新登录前cookie：{}", account.getCookie());

        // 第一步 https://labs.google/fx/api/auth/providers
        Request request = new Request.Builder()
                .url("https://labs.google/fx/api/auth/providers")
                .get()
                .addHeader("accept", "*/*")
                .addHeader("content-type", "application/json")
                .addHeader("Cookie", account.getCookie())
                .build();

        try (Response resp = CLIENT.newCall(request).execute()) {
            log.info("GET 会话请求URL: https://labs.google/fx/api/auth/providers");
            // 如需读取/更新 Cookie，可在这里处理 resp
            // 目前逻辑未用到 body，这里不读取即可，但必须关闭 resp（try-with-resources 已保证）
        }

        // 第二步 https://labs.google/fx/api/auth/csrf
        request = new Request.Builder()
                .url("https://labs.google/fx/api/auth/csrf")
                .get()
                .addHeader("User-Agent", ua)
                .addHeader("accept", "*/*")
                .addHeader("content-type", "application/json")
                .addHeader("Cookie", account.getCookie())
                .build();
        String bodyStr;

        try (Response response = CLIENT.newCall(request).execute()) {
            bodyStr = response.body() != null ? response.body().string() : null;
            log.info("GET 会话请求URL: https://labs.google/fx/api/auth/csrf, 返回数据: {}", bodyStr);
            logSetCookies(response);
        }
        JSONObject obj = JSONUtil.parseObj(bodyStr);
        String csrfToken = obj.getStr("csrfToken");

        String param = "redirect=false&csrfToken="
                + csrfToken
                + "&callbackUrl=https://labs.google/fx/tools/flow/project/"
                + account.getProjectId()
                + "&json=true";

        // 第三步 set-cookie接口 https://labs.google/fx/api/auth/signin/google
        MediaType mediaType = MediaType.parse("application/x-www-form-urlencoded");
        RequestBody body = RequestBody.create(mediaType, param);
        request = new Request.Builder()
                .url("https://labs.google/fx/api/auth/signin/google")
                .post(body)
                .addHeader("User-Agent", ua)
                .addHeader("Content-Type", "application/x-www-form-urlencoded")
                .addHeader("Cookie", account.getCookie())
                .addHeader("Accept", "*/*")
                .build();

        String step3Resp;
        try (Response response = CLIENT.newCall(request).execute()) {
            step3Resp = response.body() != null ? response.body().string() : null;
            log.info("POST 会话请求URL: https://labs.google/fx/api/auth/signin/google, 请求参数param: {}", param);
            log.info("POST 会话请求URL: https://labs.google/fx/api/auth/signin/google, 返回数据: {}", step3Resp);
            logSetCookies(response);
            CookieHelper.updateAccountCookie(account, response);
            log.info("第一次刷新cookie：{}", account.getCookie());
        }

        // 第四步 第三步返回的url
        JSONObject object = JSONUtil.parseObj(step3Resp);
        String url = object.getStr("url");
        request = new Request.Builder()
                .url(url)
                .get()
                .addHeader("User-Agent", ua)
                .addHeader("accept", "*/*")
                .addHeader("Cookie", account.getCookieFile())
                .build();

        String redirectUrl = null;
        String locationHeader;
        try (Response response = CLIENT.newCall(request).execute()) {
            String html = response.body() != null ? response.body().string() : null;

            Pattern pattern = Pattern.compile("<A HREF=\"(.*?)\">", Pattern.CASE_INSENSITIVE);
            Matcher matcher = pattern.matcher(html != null ? html : "");
            if (matcher.find()) {
                redirectUrl = matcher.group(1);
            }

            locationHeader = response.header("location");

            log.info("GET 会话请求URL: {}", url);
            log.info("返回数据: {}", html);
            log.info("返回Redirect URL: {}", redirectUrl);
            log.info("返回location: {}", locationHeader);
            logSetCookies(response);
        }

        // 第五步 set-cookie接口 第四步返回的跳转url（优先使用响应头的 location）
        String location = locationHeader;
        request = new Request.Builder()
                .url(location)
                .get()
                .addHeader("User-Agent", ua)
                .addHeader("accept", "*/*")
                .addHeader("content-type", "application/json")
                .addHeader("Cookie", account.getCookie())
                .build();

        String step5LocationHeader;
        try (Response response = CLIENT.newCall(request).execute()) {
            CookieHelper.updateAccountCookie(account, response);
            log.info("第二次刷新cookie：{}", account.getCookie());

            // 不再使用 response.body().toString()（不会消费内容且不关闭）
            // 如果你想看内容，可以按需读取：
            // String s = response.body() != null ? response.body().string() : null;
            // log.info("response body: {}", s);

            step5LocationHeader = response.header("location");
            log.info("返回location: {}", step5LocationHeader);
            logSetCookies(response);
        }

        // 第六步 set-cookie接口 项目页
        String projectUrl = "https://labs.google/fx/tools/flow/project/" + account.getProjectId();
        request = new Request.Builder()
                .url(projectUrl)
                .get()
                .addHeader("User-Agent", ua)
                .addHeader("accept", "*/*")
                .addHeader("content-type", "application/json")
                .addHeader("Cookie", account.getCookie())
                .build();

        try (Response response = CLIENT.newCall(request).execute()) {
            CookieHelper.updateAccountCookie(account, response);
            log.info("GET 会话请求URL: {}", projectUrl);
            log.info("第三次刷新cookie：{}", account.getCookie());
            logSetCookies(response);
        }
    }