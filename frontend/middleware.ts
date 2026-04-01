import { NextRequest, NextResponse } from "next/server";

function authRequired() {
  if (process.env.CSD_ALLOW_UNAUTHENTICATED === "1") {
    return false;
  }

  return (
    process.env.NODE_ENV === "production" ||
    Boolean(
      process.env.CSD_BASIC_AUTH_USERNAME &&
        process.env.CSD_BASIC_AUTH_PASSWORD
    )
  );
}

function validCredentials(username: string, password: string) {
  const expectedUser = process.env.CSD_BASIC_AUTH_USERNAME ?? "";
  const expectedPass = process.env.CSD_BASIC_AUTH_PASSWORD ?? "";

  if (!expectedUser || !expectedPass) {
    return false;
  }

  const userOk = username === expectedUser;
  const passOk = password === expectedPass;

  return userOk && passOk;
}

function unauthorizedResponse() {
  return new NextResponse("Authentication required.", {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="Clinical Support Dashboard"'
    }
  });
}

export function middleware(request: NextRequest) {
  if (!authRequired()) {
    return NextResponse.next();
  }

  if (
    !process.env.CSD_BASIC_AUTH_USERNAME ||
    !process.env.CSD_BASIC_AUTH_PASSWORD
  ) {
    return NextResponse.json(
      { error: "Application auth is not configured." },
      { status: 503 }
    );
  }

  const header = request.headers.get("authorization");
  if (!header?.startsWith("Basic ")) {
    return unauthorizedResponse();
  }

  try {
    const decoded = atob(header.slice(6));
    const separator = decoded.indexOf(":");
    const username = separator >= 0 ? decoded.slice(0, separator) : decoded;
    const password = separator >= 0 ? decoded.slice(separator + 1) : "";

    if (!validCredentials(username, password)) {
      return unauthorizedResponse();
    }
  } catch {
    return unauthorizedResponse();
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"]
};
