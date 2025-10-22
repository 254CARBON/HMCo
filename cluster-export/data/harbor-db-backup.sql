Defaulted container "database" out of: database, data-permissions-ensurer (init)
--
-- PostgreSQL database dump
--

\restrict 7AzeAKyl9UFeIRlbHi8zdDU2nagbT4cT9Lcvy2g7zdd0pY7X2aTG7lnbeTnq8Yk

-- Dumped from database version 15.14
-- Dumped by pg_dump version 15.14

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: update_update_time_at_column(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_update_time_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
  BEGIN
    NEW.update_time = NOW();
    RETURN NEW;
  END;
$$;


ALTER FUNCTION public.update_update_time_at_column() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: access; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.access (
    access_id integer NOT NULL,
    access_code character(1),
    comment character varying(30)
);


ALTER TABLE public.access OWNER TO postgres;

--
-- Name: access_access_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.access_access_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.access_access_id_seq OWNER TO postgres;

--
-- Name: access_access_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.access_access_id_seq OWNED BY public.access.access_id;


--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: artifact; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifact (
    id integer NOT NULL,
    project_id integer NOT NULL,
    repository_name character varying(255) NOT NULL,
    digest character varying(255) NOT NULL,
    type character varying(255) NOT NULL,
    pull_time timestamp without time zone,
    push_time timestamp without time zone,
    repository_id integer NOT NULL,
    media_type character varying(255) NOT NULL,
    manifest_media_type character varying(255) NOT NULL,
    size bigint,
    extra_attrs text,
    annotations jsonb,
    icon character varying(255),
    artifact_type character varying(255) NOT NULL
);


ALTER TABLE public.artifact OWNER TO postgres;

--
-- Name: artifact_accessory; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifact_accessory (
    id integer NOT NULL,
    artifact_id bigint,
    subject_artifact_id bigint,
    type character varying(256),
    size bigint,
    digest character varying(1024),
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    subject_artifact_digest character varying(1024),
    subject_artifact_repo character varying(1024)
);


ALTER TABLE public.artifact_accessory OWNER TO postgres;

--
-- Name: artifact_accessory_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.artifact_accessory_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.artifact_accessory_id_seq OWNER TO postgres;

--
-- Name: artifact_accessory_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.artifact_accessory_id_seq OWNED BY public.artifact_accessory.id;


--
-- Name: artifact_blob; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifact_blob (
    id integer NOT NULL,
    digest_af character varying(255) NOT NULL,
    digest_blob character varying(255) NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.artifact_blob OWNER TO postgres;

--
-- Name: artifact_blob_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.artifact_blob_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.artifact_blob_id_seq OWNER TO postgres;

--
-- Name: artifact_blob_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.artifact_blob_id_seq OWNED BY public.artifact_blob.id;


--
-- Name: artifact_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.artifact_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.artifact_id_seq OWNER TO postgres;

--
-- Name: artifact_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.artifact_id_seq OWNED BY public.artifact.id;


--
-- Name: artifact_reference; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifact_reference (
    id integer NOT NULL,
    parent_id integer NOT NULL,
    child_id integer NOT NULL,
    child_digest character varying(255) NOT NULL,
    platform character varying(255),
    urls character varying(1024),
    annotations jsonb
);


ALTER TABLE public.artifact_reference OWNER TO postgres;

--
-- Name: artifact_reference_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.artifact_reference_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.artifact_reference_id_seq OWNER TO postgres;

--
-- Name: artifact_reference_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.artifact_reference_id_seq OWNED BY public.artifact_reference.id;


--
-- Name: artifact_trash; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.artifact_trash (
    id integer NOT NULL,
    media_type character varying(255) NOT NULL,
    manifest_media_type character varying(255) NOT NULL,
    repository_name character varying(255) NOT NULL,
    digest character varying(255) NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.artifact_trash OWNER TO postgres;

--
-- Name: artifact_trash_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.artifact_trash_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.artifact_trash_id_seq OWNER TO postgres;

--
-- Name: artifact_trash_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.artifact_trash_id_seq OWNED BY public.artifact_trash.id;


--
-- Name: audit_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.audit_log (
    id integer NOT NULL,
    project_id integer NOT NULL,
    operation character varying(20) NOT NULL,
    resource_type character varying(255) NOT NULL,
    resource character varying(1024) NOT NULL,
    username character varying(255) NOT NULL,
    op_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.audit_log OWNER TO postgres;

--
-- Name: audit_log_ext; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.audit_log_ext (
    id bigint NOT NULL,
    project_id bigint,
    operation character varying(50),
    resource_type character varying(255),
    resource character varying(1024),
    username character varying(255),
    op_desc character varying(1024),
    op_result boolean DEFAULT true,
    payload text,
    source_ip character varying(50),
    op_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.audit_log_ext OWNER TO postgres;

--
-- Name: audit_log_ext_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.audit_log_ext_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.audit_log_ext_id_seq OWNER TO postgres;

--
-- Name: audit_log_ext_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.audit_log_ext_id_seq OWNED BY public.audit_log_ext.id;


--
-- Name: audit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.audit_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.audit_log_id_seq OWNER TO postgres;

--
-- Name: audit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.audit_log_id_seq OWNED BY public.audit_log.id;


--
-- Name: blob; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.blob (
    id integer NOT NULL,
    digest character varying(255) NOT NULL,
    content_type character varying(1024) NOT NULL,
    size bigint NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    status character varying(255) DEFAULT 'none'::character varying,
    version bigint DEFAULT 0
);


ALTER TABLE public.blob OWNER TO postgres;

--
-- Name: blob_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.blob_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.blob_id_seq OWNER TO postgres;

--
-- Name: blob_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.blob_id_seq OWNED BY public.blob.id;


--
-- Name: cve_allowlist; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cve_allowlist (
    id integer NOT NULL,
    project_id integer,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    expires_at bigint,
    items text NOT NULL
);


ALTER TABLE public.cve_allowlist OWNER TO postgres;

--
-- Name: cve_whitelist_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.cve_whitelist_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cve_whitelist_id_seq OWNER TO postgres;

--
-- Name: cve_whitelist_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.cve_whitelist_id_seq OWNED BY public.cve_allowlist.id;


--
-- Name: data_migrations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.data_migrations (
    id integer NOT NULL,
    version integer,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.data_migrations OWNER TO postgres;

--
-- Name: data_migrations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.data_migrations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.data_migrations_id_seq OWNER TO postgres;

--
-- Name: data_migrations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.data_migrations_id_seq OWNED BY public.data_migrations.id;


--
-- Name: execution; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.execution (
    id integer NOT NULL,
    vendor_type character varying(64) NOT NULL,
    vendor_id integer,
    status character varying(16),
    status_message text,
    trigger character varying(16) NOT NULL,
    extra_attrs json,
    start_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    end_time timestamp without time zone,
    revision integer,
    update_time timestamp without time zone
);


ALTER TABLE public.execution OWNER TO postgres;

--
-- Name: execution_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.execution_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.execution_id_seq OWNER TO postgres;

--
-- Name: execution_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.execution_id_seq OWNED BY public.execution.id;


--
-- Name: harbor_label; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.harbor_label (
    id integer NOT NULL,
    name character varying(128) NOT NULL,
    description text,
    color character varying(16),
    level character(1) NOT NULL,
    scope character(1) NOT NULL,
    project_id integer,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    deleted boolean DEFAULT false NOT NULL
);


ALTER TABLE public.harbor_label OWNER TO postgres;

--
-- Name: harbor_label_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.harbor_label_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.harbor_label_id_seq OWNER TO postgres;

--
-- Name: harbor_label_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.harbor_label_id_seq OWNED BY public.harbor_label.id;


--
-- Name: harbor_user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.harbor_user (
    user_id integer NOT NULL,
    username character varying(255),
    email character varying(255),
    password character varying(40) NOT NULL,
    realname character varying(255) NOT NULL,
    comment character varying(30),
    deleted boolean DEFAULT false NOT NULL,
    reset_uuid character varying(40) DEFAULT NULL::character varying,
    salt character varying(40) DEFAULT NULL::character varying,
    sysadmin_flag boolean DEFAULT false NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    password_version character varying(16) DEFAULT 'sha256'::character varying
);


ALTER TABLE public.harbor_user OWNER TO postgres;

--
-- Name: harbor_user_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.harbor_user_user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.harbor_user_user_id_seq OWNER TO postgres;

--
-- Name: harbor_user_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.harbor_user_user_id_seq OWNED BY public.harbor_user.user_id;


--
-- Name: immutable_tag_rule; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.immutable_tag_rule (
    id integer NOT NULL,
    project_id integer NOT NULL,
    tag_filter text,
    disabled boolean DEFAULT false NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.immutable_tag_rule OWNER TO postgres;

--
-- Name: immutable_tag_rule_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.immutable_tag_rule_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.immutable_tag_rule_id_seq OWNER TO postgres;

--
-- Name: immutable_tag_rule_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.immutable_tag_rule_id_seq OWNED BY public.immutable_tag_rule.id;


--
-- Name: job_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.job_log (
    log_id integer NOT NULL,
    job_uuid character varying(64) NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    content text
);


ALTER TABLE public.job_log OWNER TO postgres;

--
-- Name: job_log_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.job_log_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.job_log_log_id_seq OWNER TO postgres;

--
-- Name: job_log_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.job_log_log_id_seq OWNED BY public.job_log.log_id;


--
-- Name: job_queue_status; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.job_queue_status (
    id integer NOT NULL,
    job_type character varying(256) NOT NULL,
    paused boolean DEFAULT false NOT NULL,
    update_time timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.job_queue_status OWNER TO postgres;

--
-- Name: job_queue_status_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.job_queue_status_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.job_queue_status_id_seq OWNER TO postgres;

--
-- Name: job_queue_status_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.job_queue_status_id_seq OWNED BY public.job_queue_status.id;


--
-- Name: label_reference; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.label_reference (
    id integer NOT NULL,
    label_id integer NOT NULL,
    artifact_id integer NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.label_reference OWNER TO postgres;

--
-- Name: label_reference_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.label_reference_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.label_reference_id_seq OWNER TO postgres;

--
-- Name: label_reference_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.label_reference_id_seq OWNED BY public.label_reference.id;


--
-- Name: notification_policy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.notification_policy (
    id integer NOT NULL,
    name character varying(256),
    project_id integer NOT NULL,
    enabled boolean DEFAULT true NOT NULL,
    description text,
    targets text,
    event_types text,
    creator character varying(256),
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.notification_policy OWNER TO postgres;

--
-- Name: notification_policy_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.notification_policy_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.notification_policy_id_seq OWNER TO postgres;

--
-- Name: notification_policy_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.notification_policy_id_seq OWNED BY public.notification_policy.id;


--
-- Name: oidc_user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.oidc_user (
    id integer NOT NULL,
    user_id integer NOT NULL,
    secret character varying(255) NOT NULL,
    subiss character varying(255) NOT NULL,
    token text,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.oidc_user OWNER TO postgres;

--
-- Name: oidc_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.oidc_user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.oidc_user_id_seq OWNER TO postgres;

--
-- Name: oidc_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.oidc_user_id_seq OWNED BY public.oidc_user.id;


--
-- Name: p2p_preheat_instance; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.p2p_preheat_instance (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    description character varying(255),
    vendor character varying(255) NOT NULL,
    endpoint character varying(255) NOT NULL,
    auth_mode character varying(255),
    auth_data text,
    enabled boolean,
    is_default boolean,
    insecure boolean,
    setup_timestamp integer
);


ALTER TABLE public.p2p_preheat_instance OWNER TO postgres;

--
-- Name: p2p_preheat_instance_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.p2p_preheat_instance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.p2p_preheat_instance_id_seq OWNER TO postgres;

--
-- Name: p2p_preheat_instance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.p2p_preheat_instance_id_seq OWNED BY public.p2p_preheat_instance.id;


--
-- Name: p2p_preheat_policy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.p2p_preheat_policy (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    description character varying(1024),
    project_id integer NOT NULL,
    provider_id integer NOT NULL,
    filters character varying(1024),
    trigger character varying(255),
    enabled boolean,
    creation_time timestamp without time zone,
    update_time timestamp without time zone,
    extra_attrs text
);


ALTER TABLE public.p2p_preheat_policy OWNER TO postgres;

--
-- Name: p2p_preheat_policy_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.p2p_preheat_policy_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.p2p_preheat_policy_id_seq OWNER TO postgres;

--
-- Name: p2p_preheat_policy_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.p2p_preheat_policy_id_seq OWNED BY public.p2p_preheat_policy.id;


--
-- Name: permission_policy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.permission_policy (
    id bigint NOT NULL,
    scope character varying(255) NOT NULL,
    resource character varying(255),
    action character varying(255),
    effect character varying(255),
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.permission_policy OWNER TO postgres;

--
-- Name: permission_policy_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.permission_policy_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.permission_policy_id_seq OWNER TO postgres;

--
-- Name: permission_policy_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.permission_policy_id_seq OWNED BY public.permission_policy.id;


--
-- Name: project; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.project (
    project_id integer NOT NULL,
    owner_id integer NOT NULL,
    name character varying(255) NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    deleted boolean DEFAULT false NOT NULL,
    registry_id integer
);


ALTER TABLE public.project OWNER TO postgres;

--
-- Name: project_blob; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.project_blob (
    id integer NOT NULL,
    project_id integer NOT NULL,
    blob_id integer NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.project_blob OWNER TO postgres;

--
-- Name: project_blob_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.project_blob_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.project_blob_id_seq OWNER TO postgres;

--
-- Name: project_blob_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.project_blob_id_seq OWNED BY public.project_blob.id;


--
-- Name: project_member; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.project_member (
    id integer NOT NULL,
    project_id integer NOT NULL,
    entity_id integer NOT NULL,
    entity_type character(1) NOT NULL,
    role integer NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.project_member OWNER TO postgres;

--
-- Name: project_member_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.project_member_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.project_member_id_seq OWNER TO postgres;

--
-- Name: project_member_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.project_member_id_seq OWNED BY public.project_member.id;


--
-- Name: project_metadata; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.project_metadata (
    id integer NOT NULL,
    project_id integer NOT NULL,
    name character varying(255) NOT NULL,
    value character varying(255),
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.project_metadata OWNER TO postgres;

--
-- Name: project_metadata_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.project_metadata_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.project_metadata_id_seq OWNER TO postgres;

--
-- Name: project_metadata_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.project_metadata_id_seq OWNED BY public.project_metadata.id;


--
-- Name: project_project_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.project_project_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.project_project_id_seq OWNER TO postgres;

--
-- Name: project_project_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.project_project_id_seq OWNED BY public.project.project_id;


--
-- Name: properties; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.properties (
    id integer NOT NULL,
    k character varying(64) NOT NULL,
    v text NOT NULL
);


ALTER TABLE public.properties OWNER TO postgres;

--
-- Name: properties_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.properties_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.properties_id_seq OWNER TO postgres;

--
-- Name: properties_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.properties_id_seq OWNED BY public.properties.id;


--
-- Name: quota; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.quota (
    id integer NOT NULL,
    reference character varying(255) NOT NULL,
    reference_id character varying(255) NOT NULL,
    hard jsonb NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    version bigint DEFAULT 0
);


ALTER TABLE public.quota OWNER TO postgres;

--
-- Name: quota_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.quota_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.quota_id_seq OWNER TO postgres;

--
-- Name: quota_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.quota_id_seq OWNED BY public.quota.id;


--
-- Name: quota_usage; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.quota_usage (
    id integer NOT NULL,
    reference character varying(255) NOT NULL,
    reference_id character varying(255) NOT NULL,
    used jsonb NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    version bigint DEFAULT 0
);


ALTER TABLE public.quota_usage OWNER TO postgres;

--
-- Name: quota_usage_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.quota_usage_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.quota_usage_id_seq OWNER TO postgres;

--
-- Name: quota_usage_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.quota_usage_id_seq OWNED BY public.quota_usage.id;


--
-- Name: registry; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.registry (
    id integer NOT NULL,
    name character varying(64),
    url character varying(256),
    access_key character varying(255),
    access_secret character varying(4096),
    insecure boolean DEFAULT false NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    credential_type character varying(16),
    type character varying(32),
    description text,
    health character varying(16)
);


ALTER TABLE public.registry OWNER TO postgres;

--
-- Name: replication_policy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.replication_policy (
    id integer NOT NULL,
    name character varying(256),
    dest_registry_id integer,
    enabled boolean DEFAULT true NOT NULL,
    description text,
    deleted boolean DEFAULT false NOT NULL,
    trigger character varying(256),
    filters character varying(1024),
    replicate_deletion boolean DEFAULT false NOT NULL,
    start_time timestamp without time zone,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    creator character varying(256),
    src_registry_id integer,
    dest_namespace character varying(256),
    override boolean,
    dest_namespace_replace_count integer,
    speed_kb integer,
    copy_by_chunk boolean,
    single_active_replication boolean
);


ALTER TABLE public.replication_policy OWNER TO postgres;

--
-- Name: replication_policy_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.replication_policy_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.replication_policy_id_seq OWNER TO postgres;

--
-- Name: replication_policy_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.replication_policy_id_seq OWNED BY public.replication_policy.id;


--
-- Name: replication_target_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.replication_target_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.replication_target_id_seq OWNER TO postgres;

--
-- Name: replication_target_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.replication_target_id_seq OWNED BY public.registry.id;


--
-- Name: report_vulnerability_record; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.report_vulnerability_record (
    id bigint NOT NULL,
    report_uuid text DEFAULT ''::text NOT NULL,
    vuln_record_id bigint DEFAULT 0 NOT NULL
);


ALTER TABLE public.report_vulnerability_record OWNER TO postgres;

--
-- Name: report_vulnerability_record_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.report_vulnerability_record_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.report_vulnerability_record_id_seq OWNER TO postgres;

--
-- Name: report_vulnerability_record_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.report_vulnerability_record_id_seq OWNED BY public.report_vulnerability_record.id;


--
-- Name: repository; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.repository (
    repository_id integer NOT NULL,
    name character varying(255) NOT NULL,
    project_id integer NOT NULL,
    description text,
    pull_count integer DEFAULT 0 NOT NULL,
    star_count integer DEFAULT 0 NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.repository OWNER TO postgres;

--
-- Name: repository_repository_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.repository_repository_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.repository_repository_id_seq OWNER TO postgres;

--
-- Name: repository_repository_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.repository_repository_id_seq OWNED BY public.repository.repository_id;


--
-- Name: retention_policy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.retention_policy (
    id integer NOT NULL,
    scope_level character varying(20),
    scope_reference integer,
    trigger_kind character varying(20),
    data text,
    create_time timestamp without time zone,
    update_time timestamp without time zone
);


ALTER TABLE public.retention_policy OWNER TO postgres;

--
-- Name: retention_policy_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.retention_policy_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.retention_policy_id_seq OWNER TO postgres;

--
-- Name: retention_policy_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.retention_policy_id_seq OWNED BY public.retention_policy.id;


--
-- Name: robot; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.robot (
    id integer NOT NULL,
    name character varying(255),
    description character varying(1024),
    project_id integer,
    expiresat bigint,
    disabled boolean DEFAULT false NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    visible boolean DEFAULT true NOT NULL,
    secret character varying(2048),
    salt character varying(64),
    duration integer,
    creator_ref integer DEFAULT 0,
    creator_type character varying(255)
);


ALTER TABLE public.robot OWNER TO postgres;

--
-- Name: robot_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.robot_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.robot_id_seq OWNER TO postgres;

--
-- Name: robot_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.robot_id_seq OWNED BY public.robot.id;


--
-- Name: role; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.role (
    role_id integer NOT NULL,
    role_mask integer DEFAULT 0 NOT NULL,
    role_code character varying(20),
    name character varying(20)
);


ALTER TABLE public.role OWNER TO postgres;

--
-- Name: role_permission; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.role_permission (
    id bigint NOT NULL,
    role_type character varying(255) NOT NULL,
    role_id integer NOT NULL,
    permission_policy_id bigint NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.role_permission OWNER TO postgres;

--
-- Name: role_permission_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.role_permission_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.role_permission_id_seq OWNER TO postgres;

--
-- Name: role_permission_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.role_permission_id_seq OWNED BY public.role_permission.id;


--
-- Name: role_role_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.role_role_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.role_role_id_seq OWNER TO postgres;

--
-- Name: role_role_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.role_role_id_seq OWNED BY public.role.role_id;


--
-- Name: sbom_report; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sbom_report (
    id integer NOT NULL,
    uuid character varying(64) NOT NULL,
    artifact_id integer NOT NULL,
    registration_uuid character varying(64) NOT NULL,
    mime_type character varying(256) NOT NULL,
    media_type character varying(256) NOT NULL,
    report json
);


ALTER TABLE public.sbom_report OWNER TO postgres;

--
-- Name: sbom_report_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.sbom_report_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.sbom_report_id_seq OWNER TO postgres;

--
-- Name: sbom_report_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.sbom_report_id_seq OWNED BY public.sbom_report.id;


--
-- Name: scan_report; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scan_report (
    id integer NOT NULL,
    uuid character varying(64) NOT NULL,
    digest character varying(256) NOT NULL,
    registration_uuid character varying(64) NOT NULL,
    mime_type character varying(256) NOT NULL,
    report json,
    critical_cnt bigint,
    high_cnt bigint,
    medium_cnt bigint,
    low_cnt bigint,
    none_cnt bigint,
    unknown_cnt bigint,
    fixable_cnt bigint
);


ALTER TABLE public.scan_report OWNER TO postgres;

--
-- Name: scan_report_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.scan_report_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.scan_report_id_seq OWNER TO postgres;

--
-- Name: scan_report_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.scan_report_id_seq OWNED BY public.scan_report.id;


--
-- Name: scanner_registration; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scanner_registration (
    id integer NOT NULL,
    uuid character varying(64) NOT NULL,
    url character varying(256) NOT NULL,
    name character varying(128) NOT NULL,
    description character varying(1024),
    auth character varying(16) NOT NULL,
    access_cred character varying(512),
    disabled boolean DEFAULT false NOT NULL,
    is_default boolean DEFAULT false NOT NULL,
    use_internal_addr boolean DEFAULT false NOT NULL,
    immutable boolean DEFAULT false NOT NULL,
    skip_cert_verify boolean DEFAULT false NOT NULL,
    create_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.scanner_registration OWNER TO postgres;

--
-- Name: scanner_registration_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.scanner_registration_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.scanner_registration_id_seq OWNER TO postgres;

--
-- Name: scanner_registration_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.scanner_registration_id_seq OWNED BY public.scanner_registration.id;


--
-- Name: schedule; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.schedule (
    id integer NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    vendor_type character varying(64),
    vendor_id integer,
    cron character varying(64),
    callback_func_name character varying(128),
    callback_func_param text,
    cron_type character varying(64),
    extra_attrs json,
    revision integer
);


ALTER TABLE public.schedule OWNER TO postgres;

--
-- Name: schedule_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.schedule_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.schedule_id_seq OWNER TO postgres;

--
-- Name: schedule_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.schedule_id_seq OWNED BY public.schedule.id;


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.schema_migrations (
    version bigint NOT NULL,
    dirty boolean NOT NULL
);


ALTER TABLE public.schema_migrations OWNER TO postgres;

--
-- Name: system_artifact; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.system_artifact (
    id integer NOT NULL,
    repository character varying(256) NOT NULL,
    digest character varying(255) DEFAULT ''::character varying NOT NULL,
    size bigint DEFAULT 0 NOT NULL,
    vendor character varying(255) DEFAULT ''::character varying NOT NULL,
    type character varying(255) DEFAULT ''::character varying NOT NULL,
    create_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    extra_attrs text DEFAULT ''::text NOT NULL
);


ALTER TABLE public.system_artifact OWNER TO postgres;

--
-- Name: system_artifact_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.system_artifact_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.system_artifact_id_seq OWNER TO postgres;

--
-- Name: system_artifact_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.system_artifact_id_seq OWNED BY public.system_artifact.id;


--
-- Name: tag; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tag (
    id integer NOT NULL,
    repository_id integer NOT NULL,
    artifact_id integer NOT NULL,
    name character varying(255) NOT NULL,
    push_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    pull_time timestamp without time zone
);


ALTER TABLE public.tag OWNER TO postgres;

--
-- Name: tag_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.tag_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.tag_id_seq OWNER TO postgres;

--
-- Name: tag_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.tag_id_seq OWNED BY public.tag.id;


--
-- Name: task; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.task (
    id integer NOT NULL,
    execution_id integer NOT NULL,
    job_id character varying(64),
    status character varying(16) NOT NULL,
    status_code integer NOT NULL,
    status_revision integer,
    status_message text,
    run_count integer,
    extra_attrs json,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    start_time timestamp without time zone,
    update_time timestamp without time zone,
    end_time timestamp without time zone,
    vendor_type character varying(64) NOT NULL
);


ALTER TABLE public.task OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.task_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.task_id_seq OWNER TO postgres;

--
-- Name: task_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.task_id_seq OWNED BY public.task.id;


--
-- Name: user_group; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_group (
    id integer NOT NULL,
    group_name character varying(255) NOT NULL,
    group_type smallint DEFAULT 0,
    ldap_group_dn character varying(512) NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    update_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.user_group OWNER TO postgres;

--
-- Name: user_group_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_group_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_group_id_seq OWNER TO postgres;

--
-- Name: user_group_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_group_id_seq OWNED BY public.user_group.id;


--
-- Name: vulnerability_record; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vulnerability_record (
    id bigint NOT NULL,
    cve_id text DEFAULT ''::text NOT NULL,
    registration_uuid text DEFAULT ''::text NOT NULL,
    package text DEFAULT ''::text NOT NULL,
    package_version text DEFAULT ''::text NOT NULL,
    package_type text DEFAULT ''::text NOT NULL,
    severity text DEFAULT ''::text NOT NULL,
    fixed_version text,
    urls text,
    cvss_score_v3 double precision,
    cvss_score_v2 double precision,
    cvss_vector_v3 text,
    cvss_vector_v2 text,
    description text,
    cwe_ids text,
    vendor_attributes json,
    status text
);


ALTER TABLE public.vulnerability_record OWNER TO postgres;

--
-- Name: vulnerability_record_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.vulnerability_record_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.vulnerability_record_id_seq OWNER TO postgres;

--
-- Name: vulnerability_record_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.vulnerability_record_id_seq OWNED BY public.vulnerability_record.id;


--
-- Name: access access_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.access ALTER COLUMN access_id SET DEFAULT nextval('public.access_access_id_seq'::regclass);


--
-- Name: artifact id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact ALTER COLUMN id SET DEFAULT nextval('public.artifact_id_seq'::regclass);


--
-- Name: artifact_accessory id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_accessory ALTER COLUMN id SET DEFAULT nextval('public.artifact_accessory_id_seq'::regclass);


--
-- Name: artifact_blob id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_blob ALTER COLUMN id SET DEFAULT nextval('public.artifact_blob_id_seq'::regclass);


--
-- Name: artifact_reference id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_reference ALTER COLUMN id SET DEFAULT nextval('public.artifact_reference_id_seq'::regclass);


--
-- Name: artifact_trash id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_trash ALTER COLUMN id SET DEFAULT nextval('public.artifact_trash_id_seq'::regclass);


--
-- Name: audit_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log ALTER COLUMN id SET DEFAULT nextval('public.audit_log_id_seq'::regclass);


--
-- Name: audit_log_ext id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log_ext ALTER COLUMN id SET DEFAULT nextval('public.audit_log_ext_id_seq'::regclass);


--
-- Name: blob id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.blob ALTER COLUMN id SET DEFAULT nextval('public.blob_id_seq'::regclass);


--
-- Name: cve_allowlist id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cve_allowlist ALTER COLUMN id SET DEFAULT nextval('public.cve_whitelist_id_seq'::regclass);


--
-- Name: data_migrations id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_migrations ALTER COLUMN id SET DEFAULT nextval('public.data_migrations_id_seq'::regclass);


--
-- Name: execution id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.execution ALTER COLUMN id SET DEFAULT nextval('public.execution_id_seq'::regclass);


--
-- Name: harbor_label id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_label ALTER COLUMN id SET DEFAULT nextval('public.harbor_label_id_seq'::regclass);


--
-- Name: harbor_user user_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_user ALTER COLUMN user_id SET DEFAULT nextval('public.harbor_user_user_id_seq'::regclass);


--
-- Name: immutable_tag_rule id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.immutable_tag_rule ALTER COLUMN id SET DEFAULT nextval('public.immutable_tag_rule_id_seq'::regclass);


--
-- Name: job_log log_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.job_log ALTER COLUMN log_id SET DEFAULT nextval('public.job_log_log_id_seq'::regclass);


--
-- Name: job_queue_status id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.job_queue_status ALTER COLUMN id SET DEFAULT nextval('public.job_queue_status_id_seq'::regclass);


--
-- Name: label_reference id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.label_reference ALTER COLUMN id SET DEFAULT nextval('public.label_reference_id_seq'::regclass);


--
-- Name: notification_policy id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notification_policy ALTER COLUMN id SET DEFAULT nextval('public.notification_policy_id_seq'::regclass);


--
-- Name: oidc_user id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.oidc_user ALTER COLUMN id SET DEFAULT nextval('public.oidc_user_id_seq'::regclass);


--
-- Name: p2p_preheat_instance id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.p2p_preheat_instance ALTER COLUMN id SET DEFAULT nextval('public.p2p_preheat_instance_id_seq'::regclass);


--
-- Name: p2p_preheat_policy id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.p2p_preheat_policy ALTER COLUMN id SET DEFAULT nextval('public.p2p_preheat_policy_id_seq'::regclass);


--
-- Name: permission_policy id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.permission_policy ALTER COLUMN id SET DEFAULT nextval('public.permission_policy_id_seq'::regclass);


--
-- Name: project project_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project ALTER COLUMN project_id SET DEFAULT nextval('public.project_project_id_seq'::regclass);


--
-- Name: project_blob id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_blob ALTER COLUMN id SET DEFAULT nextval('public.project_blob_id_seq'::regclass);


--
-- Name: project_member id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_member ALTER COLUMN id SET DEFAULT nextval('public.project_member_id_seq'::regclass);


--
-- Name: project_metadata id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_metadata ALTER COLUMN id SET DEFAULT nextval('public.project_metadata_id_seq'::regclass);


--
-- Name: properties id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.properties ALTER COLUMN id SET DEFAULT nextval('public.properties_id_seq'::regclass);


--
-- Name: quota id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.quota ALTER COLUMN id SET DEFAULT nextval('public.quota_id_seq'::regclass);


--
-- Name: quota_usage id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.quota_usage ALTER COLUMN id SET DEFAULT nextval('public.quota_usage_id_seq'::regclass);


--
-- Name: registry id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.registry ALTER COLUMN id SET DEFAULT nextval('public.replication_target_id_seq'::regclass);


--
-- Name: replication_policy id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.replication_policy ALTER COLUMN id SET DEFAULT nextval('public.replication_policy_id_seq'::regclass);


--
-- Name: report_vulnerability_record id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report_vulnerability_record ALTER COLUMN id SET DEFAULT nextval('public.report_vulnerability_record_id_seq'::regclass);


--
-- Name: repository repository_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.repository ALTER COLUMN repository_id SET DEFAULT nextval('public.repository_repository_id_seq'::regclass);


--
-- Name: retention_policy id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.retention_policy ALTER COLUMN id SET DEFAULT nextval('public.retention_policy_id_seq'::regclass);


--
-- Name: robot id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.robot ALTER COLUMN id SET DEFAULT nextval('public.robot_id_seq'::regclass);


--
-- Name: role role_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.role ALTER COLUMN role_id SET DEFAULT nextval('public.role_role_id_seq'::regclass);


--
-- Name: role_permission id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.role_permission ALTER COLUMN id SET DEFAULT nextval('public.role_permission_id_seq'::regclass);


--
-- Name: sbom_report id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sbom_report ALTER COLUMN id SET DEFAULT nextval('public.sbom_report_id_seq'::regclass);


--
-- Name: scan_report id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scan_report ALTER COLUMN id SET DEFAULT nextval('public.scan_report_id_seq'::regclass);


--
-- Name: scanner_registration id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_registration ALTER COLUMN id SET DEFAULT nextval('public.scanner_registration_id_seq'::regclass);


--
-- Name: schedule id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedule ALTER COLUMN id SET DEFAULT nextval('public.schedule_id_seq'::regclass);


--
-- Name: system_artifact id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.system_artifact ALTER COLUMN id SET DEFAULT nextval('public.system_artifact_id_seq'::regclass);


--
-- Name: tag id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag ALTER COLUMN id SET DEFAULT nextval('public.tag_id_seq'::regclass);


--
-- Name: task id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task ALTER COLUMN id SET DEFAULT nextval('public.task_id_seq'::regclass);


--
-- Name: user_group id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_group ALTER COLUMN id SET DEFAULT nextval('public.user_group_id_seq'::regclass);


--
-- Name: vulnerability_record id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vulnerability_record ALTER COLUMN id SET DEFAULT nextval('public.vulnerability_record_id_seq'::regclass);


--
-- Data for Name: access; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.access (access_id, access_code, comment) FROM stdin;
1	M	Management access for project
2	R	Read access for project
3	W	Write access for project
4	D	Delete access for project
5	S	Search access for project
\.


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
1.6.0
\.


--
-- Data for Name: artifact; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.artifact (id, project_id, repository_name, digest, type, pull_time, push_time, repository_id, media_type, manifest_media_type, size, extra_attrs, annotations, icon, artifact_type) FROM stdin;
\.


--
-- Data for Name: artifact_accessory; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.artifact_accessory (id, artifact_id, subject_artifact_id, type, size, digest, creation_time, subject_artifact_digest, subject_artifact_repo) FROM stdin;
\.


--
-- Data for Name: artifact_blob; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.artifact_blob (id, digest_af, digest_blob, creation_time) FROM stdin;
\.


--
-- Data for Name: artifact_reference; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.artifact_reference (id, parent_id, child_id, child_digest, platform, urls, annotations) FROM stdin;
\.


--
-- Data for Name: artifact_trash; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.artifact_trash (id, media_type, manifest_media_type, repository_name, digest, creation_time) FROM stdin;
\.


--
-- Data for Name: audit_log; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.audit_log (id, project_id, operation, resource_type, resource, username, op_time) FROM stdin;
\.


--
-- Data for Name: audit_log_ext; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.audit_log_ext (id, project_id, operation, resource_type, resource, username, op_desc, op_result, payload, source_ip, op_time) FROM stdin;
1	2	create	project	254carbon	admin	create project: 254carbon	t	\N	\N	2025-10-20 15:24:39.44001
2	2	create	robot	robot$254carbon+robot-ci	admin	create robot: robot$254carbon+robot-ci	t	\N	\N	2025-10-20 15:26:24.326885
\.


--
-- Data for Name: blob; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.blob (id, digest, content_type, size, creation_time, update_time, status, version) FROM stdin;
\.


--
-- Data for Name: cve_allowlist; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cve_allowlist (id, project_id, creation_time, update_time, expires_at, items) FROM stdin;
1	1	2025-10-20 04:28:57.214689	2025-10-20 04:28:57.214689	\N	[]
2	2	2025-10-20 15:24:39.431023	2025-10-20 15:24:39.431023	\N	[]
\.


--
-- Data for Name: data_migrations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.data_migrations (id, version, creation_time, update_time) FROM stdin;
1	0	2025-10-20 04:28:57.17776	2025-10-20 04:28:57.17776
\.


--
-- Data for Name: execution; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.execution (id, vendor_type, vendor_id, status, status_message, trigger, extra_attrs, start_time, end_time, revision, update_time) FROM stdin;
2	SCHEDULER	2	Running		MANUAL	{}	2025-10-20 04:30:04.26564	\N	1	2025-10-20 04:34:57
1	SCHEDULER	1	Running		MANUAL	{}	2025-10-20 04:30:04.248006	\N	1	2025-10-20 04:34:57
\.


--
-- Data for Name: harbor_label; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.harbor_label (id, name, description, color, level, scope, project_id, creation_time, update_time, deleted) FROM stdin;
\.


--
-- Data for Name: harbor_user; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.harbor_user (user_id, username, email, password, realname, comment, deleted, reset_uuid, salt, sysadmin_flag, creation_time, update_time, password_version) FROM stdin;
2	anonymous	\N		anonymous user	anonymous user	t	\N	\N	f	2025-10-20 04:28:56.895491	2025-10-20 04:28:57.096976	sha1
1	admin	\N	6825bc5c9b06169b744d76f574978a30	system admin	admin user	f	\N	d1Zib0Y7CNoOclheFz8fY2ctj9OEWkFZ	t	2025-10-20 04:28:56.895491	2025-10-20 04:28:57.501562	sha256
\.


--
-- Data for Name: immutable_tag_rule; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.immutable_tag_rule (id, project_id, tag_filter, disabled, creation_time) FROM stdin;
\.


--
-- Data for Name: job_log; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.job_log (log_id, job_uuid, creation_time, content) FROM stdin;
\.


--
-- Data for Name: job_queue_status; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.job_queue_status (id, job_type, paused, update_time) FROM stdin;
\.


--
-- Data for Name: label_reference; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.label_reference (id, label_id, artifact_id, creation_time, update_time) FROM stdin;
\.


--
-- Data for Name: notification_policy; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.notification_policy (id, name, project_id, enabled, description, targets, event_types, creator, creation_time, update_time) FROM stdin;
\.


--
-- Data for Name: oidc_user; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.oidc_user (id, user_id, secret, subiss, token, creation_time, update_time) FROM stdin;
\.


--
-- Data for Name: p2p_preheat_instance; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.p2p_preheat_instance (id, name, description, vendor, endpoint, auth_mode, auth_data, enabled, is_default, insecure, setup_timestamp) FROM stdin;
\.


--
-- Data for Name: p2p_preheat_policy; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.p2p_preheat_policy (id, name, description, project_id, provider_id, filters, trigger, enabled, creation_time, update_time, extra_attrs) FROM stdin;
\.


--
-- Data for Name: permission_policy; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.permission_policy (id, scope, resource, action, effect, creation_time) FROM stdin;
1	/project/2	repository	push		2025-10-20 15:26:24.318631
2	/project/2	repository	pull		2025-10-20 15:26:24.323976
\.


--
-- Data for Name: project; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.project (project_id, owner_id, name, creation_time, update_time, deleted, registry_id) FROM stdin;
1	1	library	2025-10-20 04:28:56.895491	2025-10-20 04:28:56.895491	f	\N
2	1	254carbon	2025-10-20 15:24:39.428084	2025-10-20 15:24:39.428084	f	0
\.


--
-- Data for Name: project_blob; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.project_blob (id, project_id, blob_id, creation_time) FROM stdin;
\.


--
-- Data for Name: project_member; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.project_member (id, project_id, entity_id, entity_type, role, creation_time, update_time) FROM stdin;
1	1	1	u	1	2025-10-20 04:28:56.895491	2025-10-20 04:28:56.895491
2	2	1	u	1	2025-10-20 15:24:39.428084	2025-10-20 15:24:39.428084
\.


--
-- Data for Name: project_metadata; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.project_metadata (id, project_id, name, value, creation_time, update_time) FROM stdin;
1	1	public	true	2025-10-20 04:28:56.895491	2025-10-20 04:28:56.895491
2	2	public	false	2025-10-20 15:24:39.43249	2025-10-20 15:24:39.43249
\.


--
-- Data for Name: properties; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.properties (id, k, v) FROM stdin;
\.


--
-- Data for Name: quota; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.quota (id, reference, reference_id, hard, creation_time, update_time, version) FROM stdin;
1	project	1	{"storage": -1}	2025-10-20 04:28:57.048989	2025-10-20 04:28:57.048989	0
2	project	2	{"storage": -1}	2025-10-20 15:24:39.434557	2025-10-20 15:24:39.434557	0
\.


--
-- Data for Name: quota_usage; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.quota_usage (id, reference, reference_id, used, creation_time, update_time, version) FROM stdin;
1	project	1	{"storage": 0}	2025-10-20 04:28:57.048989	2025-10-20 04:28:57.048989	0
2	project	2	{"storage": 0}	2025-10-20 15:24:39.434557	2025-10-20 15:24:39.434557	0
\.


--
-- Data for Name: registry; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.registry (id, name, url, access_key, access_secret, insecure, creation_time, update_time, credential_type, type, description, health) FROM stdin;
\.


--
-- Data for Name: replication_policy; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.replication_policy (id, name, dest_registry_id, enabled, description, deleted, trigger, filters, replicate_deletion, start_time, creation_time, update_time, creator, src_registry_id, dest_namespace, override, dest_namespace_replace_count, speed_kb, copy_by_chunk, single_active_replication) FROM stdin;
\.


--
-- Data for Name: report_vulnerability_record; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.report_vulnerability_record (id, report_uuid, vuln_record_id) FROM stdin;
\.


--
-- Data for Name: repository; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.repository (repository_id, name, project_id, description, pull_count, star_count, creation_time, update_time) FROM stdin;
\.


--
-- Data for Name: retention_policy; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.retention_policy (id, scope_level, scope_reference, trigger_kind, data, create_time, update_time) FROM stdin;
\.


--
-- Data for Name: robot; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.robot (id, name, description, project_id, expiresat, disabled, creation_time, update_time, visible, secret, salt, duration, creator_ref, creator_type) FROM stdin;
1	254carbon+robot-ci		2	-1	f	2025-10-20 15:26:24.316697	2025-10-20 15:26:24.316719	t	93f4d6e8328fe0ff478b11eeecfc3fc5	a2WY7Nb2Xa7UUtJPgR0BJdcj4AhVVp5W	-1	1	local
\.


--
-- Data for Name: role; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.role (role_id, role_mask, role_code, name) FROM stdin;
1	0	MDRWS	projectAdmin
2	0	RWS	developer
3	0	RS	guest
5	0	LRS	limitedGuest
4	0	DRWS	maintainer
\.


--
-- Data for Name: role_permission; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.role_permission (id, role_type, role_id, permission_policy_id, creation_time) FROM stdin;
1	robotaccount	1	1	2025-10-20 15:26:24.32027
2	robotaccount	1	2	2025-10-20 15:26:24.324371
\.


--
-- Data for Name: sbom_report; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sbom_report (id, uuid, artifact_id, registration_uuid, mime_type, media_type, report) FROM stdin;
\.


--
-- Data for Name: scan_report; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.scan_report (id, uuid, digest, registration_uuid, mime_type, report, critical_cnt, high_cnt, medium_cnt, low_cnt, none_cnt, unknown_cnt, fixable_cnt) FROM stdin;
\.


--
-- Data for Name: scanner_registration; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.scanner_registration (id, uuid, url, name, description, auth, access_cred, disabled, is_default, use_internal_addr, immutable, skip_cert_verify, create_time, update_time) FROM stdin;
1	4b52a000-ad6d-11f0-bd23-ba508c2318ed	http://harbor-trivy:8080	Trivy	The Trivy scanner adapter			f	t	t	t	f	2025-10-20 04:28:57.523693	2025-10-20 04:28:57.523696
\.


--
-- Data for Name: schedule; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.schedule (id, creation_time, update_time, vendor_type, vendor_id, cron, callback_func_name, callback_func_param, cron_type, extra_attrs, revision) FROM stdin;
1	2025-10-20 04:30:04.246366	2025-10-20 04:30:04.246366	SYSTEM_ARTIFACT_CLEANUP	0	0 0 0 * * *	SYSTEM_ARTIFACT_CLEANUP	null	Daily	null	0
2	2025-10-20 04:30:04.265217	2025-10-20 04:30:04.265217	EXECUTION_SWEEP	-1	0 0 0 * * *	EXECUTION_SWEEP_CALLBACK	null	Custom	null	0
\.


--
-- Data for Name: schema_migrations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.schema_migrations (version, dirty) FROM stdin;
170	f
\.


--
-- Data for Name: system_artifact; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.system_artifact (id, repository, digest, size, vendor, type, create_time, extra_attrs) FROM stdin;
\.


--
-- Data for Name: tag; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.tag (id, repository_id, artifact_id, name, push_time, pull_time) FROM stdin;
\.


--
-- Data for Name: task; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.task (id, execution_id, job_id, status, status_code, status_revision, status_message, run_count, extra_attrs, creation_time, start_time, update_time, end_time, vendor_type) FROM stdin;
1	1	fccc7e86cff38b5235ccab47	Scheduled	1	1760934604		1	{}	2025-10-20 04:30:04.25063	2025-10-20 04:30:04	2025-10-20 04:34:32	0001-01-01 00:00:00	SCHEDULER
2	2	58489f4deb0cb895e00758df	Scheduled	1	1760934604		1	{}	2025-10-20 04:30:04.266208	2025-10-20 04:30:04	2025-10-20 04:34:32	0001-01-01 00:00:00	SCHEDULER
\.


--
-- Data for Name: user_group; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.user_group (id, group_name, group_type, ldap_group_dn, creation_time, update_time) FROM stdin;
\.


--
-- Data for Name: vulnerability_record; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vulnerability_record (id, cve_id, registration_uuid, package, package_version, package_type, severity, fixed_version, urls, cvss_score_v3, cvss_score_v2, cvss_vector_v3, cvss_vector_v2, description, cwe_ids, vendor_attributes, status) FROM stdin;
\.


--
-- Name: access_access_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.access_access_id_seq', 5, true);


--
-- Name: artifact_accessory_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.artifact_accessory_id_seq', 1, false);


--
-- Name: artifact_blob_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.artifact_blob_id_seq', 1, false);


--
-- Name: artifact_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.artifact_id_seq', 1, false);


--
-- Name: artifact_reference_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.artifact_reference_id_seq', 1, false);


--
-- Name: artifact_trash_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.artifact_trash_id_seq', 1, false);


--
-- Name: audit_log_ext_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.audit_log_ext_id_seq', 2, true);


--
-- Name: audit_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.audit_log_id_seq', 1, false);


--
-- Name: blob_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.blob_id_seq', 1, false);


--
-- Name: cve_whitelist_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.cve_whitelist_id_seq', 2, true);


--
-- Name: data_migrations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.data_migrations_id_seq', 1, true);


--
-- Name: execution_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.execution_id_seq', 2, true);


--
-- Name: harbor_label_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.harbor_label_id_seq', 1, false);


--
-- Name: harbor_user_user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.harbor_user_user_id_seq', 2, true);


--
-- Name: immutable_tag_rule_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.immutable_tag_rule_id_seq', 1, false);


--
-- Name: job_log_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.job_log_log_id_seq', 1, false);


--
-- Name: job_queue_status_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.job_queue_status_id_seq', 1, false);


--
-- Name: label_reference_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.label_reference_id_seq', 1, false);


--
-- Name: notification_policy_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.notification_policy_id_seq', 1, false);


--
-- Name: oidc_user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.oidc_user_id_seq', 1, false);


--
-- Name: p2p_preheat_instance_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.p2p_preheat_instance_id_seq', 1, false);


--
-- Name: p2p_preheat_policy_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.p2p_preheat_policy_id_seq', 1, false);


--
-- Name: permission_policy_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.permission_policy_id_seq', 2, true);


--
-- Name: project_blob_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.project_blob_id_seq', 1, false);


--
-- Name: project_member_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.project_member_id_seq', 2, true);


--
-- Name: project_metadata_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.project_metadata_id_seq', 2, true);


--
-- Name: project_project_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.project_project_id_seq', 2, true);


--
-- Name: properties_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.properties_id_seq', 1, false);


--
-- Name: quota_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.quota_id_seq', 2, true);


--
-- Name: quota_usage_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.quota_usage_id_seq', 2, true);


--
-- Name: replication_policy_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.replication_policy_id_seq', 1, false);


--
-- Name: replication_target_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.replication_target_id_seq', 1, false);


--
-- Name: report_vulnerability_record_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.report_vulnerability_record_id_seq', 1, false);


--
-- Name: repository_repository_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.repository_repository_id_seq', 1, false);


--
-- Name: retention_policy_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.retention_policy_id_seq', 1, false);


--
-- Name: robot_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.robot_id_seq', 1, true);


--
-- Name: role_permission_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.role_permission_id_seq', 2, true);


--
-- Name: role_role_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.role_role_id_seq', 5, true);


--
-- Name: sbom_report_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.sbom_report_id_seq', 1, false);


--
-- Name: scan_report_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.scan_report_id_seq', 1, false);


--
-- Name: scanner_registration_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.scanner_registration_id_seq', 1, true);


--
-- Name: schedule_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.schedule_id_seq', 2, true);


--
-- Name: system_artifact_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.system_artifact_id_seq', 1, false);


--
-- Name: tag_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.tag_id_seq', 1, false);


--
-- Name: task_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.task_id_seq', 2, true);


--
-- Name: user_group_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.user_group_id_seq', 1, false);


--
-- Name: vulnerability_record_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.vulnerability_record_id_seq', 1, false);


--
-- Name: access access_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.access
    ADD CONSTRAINT access_pkey PRIMARY KEY (access_id);


--
-- Name: artifact_accessory artifact_accessory_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_accessory
    ADD CONSTRAINT artifact_accessory_pkey PRIMARY KEY (id);


--
-- Name: artifact_blob artifact_blob_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_blob
    ADD CONSTRAINT artifact_blob_pkey PRIMARY KEY (id);


--
-- Name: artifact artifact_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact
    ADD CONSTRAINT artifact_pkey PRIMARY KEY (id);


--
-- Name: artifact_reference artifact_reference_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_reference
    ADD CONSTRAINT artifact_reference_pkey PRIMARY KEY (id);


--
-- Name: artifact_trash artifact_trash_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_trash
    ADD CONSTRAINT artifact_trash_pkey PRIMARY KEY (id);


--
-- Name: audit_log_ext audit_log_ext_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log_ext
    ADD CONSTRAINT audit_log_ext_pkey PRIMARY KEY (id);


--
-- Name: audit_log audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_log
    ADD CONSTRAINT audit_log_pkey PRIMARY KEY (id);


--
-- Name: blob blob_digest_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.blob
    ADD CONSTRAINT blob_digest_key UNIQUE (digest);


--
-- Name: blob blob_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.blob
    ADD CONSTRAINT blob_pkey PRIMARY KEY (id);


--
-- Name: cve_allowlist cve_whitelist_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cve_allowlist
    ADD CONSTRAINT cve_whitelist_pkey PRIMARY KEY (id);


--
-- Name: cve_allowlist cve_whitelist_project_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cve_allowlist
    ADD CONSTRAINT cve_whitelist_project_id_key UNIQUE (project_id);


--
-- Name: data_migrations data_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.data_migrations
    ADD CONSTRAINT data_migrations_pkey PRIMARY KEY (id);


--
-- Name: execution execution_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.execution
    ADD CONSTRAINT execution_pkey PRIMARY KEY (id);


--
-- Name: harbor_label harbor_label_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_label
    ADD CONSTRAINT harbor_label_pkey PRIMARY KEY (id);


--
-- Name: harbor_user harbor_user_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_user
    ADD CONSTRAINT harbor_user_email_key UNIQUE (email);


--
-- Name: harbor_user harbor_user_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_user
    ADD CONSTRAINT harbor_user_pkey PRIMARY KEY (user_id);


--
-- Name: harbor_user harbor_user_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_user
    ADD CONSTRAINT harbor_user_username_key UNIQUE (username);


--
-- Name: immutable_tag_rule immutable_tag_rule_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.immutable_tag_rule
    ADD CONSTRAINT immutable_tag_rule_pkey PRIMARY KEY (id);


--
-- Name: immutable_tag_rule immutable_tag_rule_project_id_tag_filter_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.immutable_tag_rule
    ADD CONSTRAINT immutable_tag_rule_project_id_tag_filter_key UNIQUE (project_id, tag_filter);


--
-- Name: job_log job_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.job_log
    ADD CONSTRAINT job_log_pkey PRIMARY KEY (log_id);


--
-- Name: job_queue_status job_queue_status_job_type_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.job_queue_status
    ADD CONSTRAINT job_queue_status_job_type_key UNIQUE (job_type);


--
-- Name: job_queue_status job_queue_status_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.job_queue_status
    ADD CONSTRAINT job_queue_status_pkey PRIMARY KEY (id);


--
-- Name: label_reference label_reference_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.label_reference
    ADD CONSTRAINT label_reference_pkey PRIMARY KEY (id);


--
-- Name: notification_policy notification_policy_name_project_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notification_policy
    ADD CONSTRAINT notification_policy_name_project_id_key UNIQUE (name, project_id);


--
-- Name: notification_policy notification_policy_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notification_policy
    ADD CONSTRAINT notification_policy_pkey PRIMARY KEY (id);


--
-- Name: oidc_user oidc_user_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.oidc_user
    ADD CONSTRAINT oidc_user_pkey PRIMARY KEY (id);


--
-- Name: oidc_user oidc_user_subiss_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.oidc_user
    ADD CONSTRAINT oidc_user_subiss_key UNIQUE (subiss);


--
-- Name: p2p_preheat_instance p2p_preheat_instance_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.p2p_preheat_instance
    ADD CONSTRAINT p2p_preheat_instance_name_key UNIQUE (name);


--
-- Name: p2p_preheat_instance p2p_preheat_instance_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.p2p_preheat_instance
    ADD CONSTRAINT p2p_preheat_instance_pkey PRIMARY KEY (id);


--
-- Name: p2p_preheat_policy p2p_preheat_policy_name_project_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.p2p_preheat_policy
    ADD CONSTRAINT p2p_preheat_policy_name_project_id_key UNIQUE (name, project_id);


--
-- Name: p2p_preheat_policy p2p_preheat_policy_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.p2p_preheat_policy
    ADD CONSTRAINT p2p_preheat_policy_pkey PRIMARY KEY (id);


--
-- Name: permission_policy permission_policy_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.permission_policy
    ADD CONSTRAINT permission_policy_pkey PRIMARY KEY (id);


--
-- Name: project_blob project_blob_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_blob
    ADD CONSTRAINT project_blob_pkey PRIMARY KEY (id);


--
-- Name: project_member project_member_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_member
    ADD CONSTRAINT project_member_pkey PRIMARY KEY (id);


--
-- Name: project_metadata project_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_metadata
    ADD CONSTRAINT project_metadata_pkey PRIMARY KEY (id);


--
-- Name: project project_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project
    ADD CONSTRAINT project_name_key UNIQUE (name);


--
-- Name: project project_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project
    ADD CONSTRAINT project_pkey PRIMARY KEY (project_id);


--
-- Name: properties properties_k_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.properties
    ADD CONSTRAINT properties_k_key UNIQUE (k);


--
-- Name: properties properties_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.properties
    ADD CONSTRAINT properties_pkey PRIMARY KEY (id);


--
-- Name: quota quota_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.quota
    ADD CONSTRAINT quota_pkey PRIMARY KEY (id);


--
-- Name: quota quota_reference_reference_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.quota
    ADD CONSTRAINT quota_reference_reference_id_key UNIQUE (reference, reference_id);


--
-- Name: quota_usage quota_usage_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.quota_usage
    ADD CONSTRAINT quota_usage_pkey PRIMARY KEY (id);


--
-- Name: quota_usage quota_usage_reference_reference_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.quota_usage
    ADD CONSTRAINT quota_usage_reference_reference_id_key UNIQUE (reference, reference_id);


--
-- Name: replication_policy replication_policy_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.replication_policy
    ADD CONSTRAINT replication_policy_pkey PRIMARY KEY (id);


--
-- Name: registry replication_target_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.registry
    ADD CONSTRAINT replication_target_pkey PRIMARY KEY (id);


--
-- Name: report_vulnerability_record report_vulnerability_record_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report_vulnerability_record
    ADD CONSTRAINT report_vulnerability_record_pkey PRIMARY KEY (id);


--
-- Name: report_vulnerability_record report_vulnerability_record_report_uuid_vuln_record_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report_vulnerability_record
    ADD CONSTRAINT report_vulnerability_record_report_uuid_vuln_record_id_key UNIQUE (report_uuid, vuln_record_id);


--
-- Name: repository repository_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.repository
    ADD CONSTRAINT repository_name_key UNIQUE (name);


--
-- Name: repository repository_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.repository
    ADD CONSTRAINT repository_pkey PRIMARY KEY (repository_id);


--
-- Name: retention_policy retention_policy_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.retention_policy
    ADD CONSTRAINT retention_policy_pkey PRIMARY KEY (id);


--
-- Name: robot robot_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.robot
    ADD CONSTRAINT robot_pkey PRIMARY KEY (id);


--
-- Name: role_permission role_permission_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.role_permission
    ADD CONSTRAINT role_permission_pkey PRIMARY KEY (id);


--
-- Name: role role_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.role
    ADD CONSTRAINT role_pkey PRIMARY KEY (role_id);


--
-- Name: sbom_report sbom_report_artifact_id_registration_uuid_mime_type_media_t_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sbom_report
    ADD CONSTRAINT sbom_report_artifact_id_registration_uuid_mime_type_media_t_key UNIQUE (artifact_id, registration_uuid, mime_type, media_type);


--
-- Name: sbom_report sbom_report_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sbom_report
    ADD CONSTRAINT sbom_report_pkey PRIMARY KEY (id);


--
-- Name: sbom_report sbom_report_uuid_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sbom_report
    ADD CONSTRAINT sbom_report_uuid_key UNIQUE (uuid);


--
-- Name: scan_report scan_report_digest_registration_uuid_mime_type_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scan_report
    ADD CONSTRAINT scan_report_digest_registration_uuid_mime_type_key UNIQUE (digest, registration_uuid, mime_type);


--
-- Name: scan_report scan_report_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scan_report
    ADD CONSTRAINT scan_report_pkey PRIMARY KEY (id);


--
-- Name: scan_report scan_report_uuid_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scan_report
    ADD CONSTRAINT scan_report_uuid_key UNIQUE (uuid);


--
-- Name: scanner_registration scanner_registration_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_registration
    ADD CONSTRAINT scanner_registration_name_key UNIQUE (name);


--
-- Name: scanner_registration scanner_registration_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_registration
    ADD CONSTRAINT scanner_registration_pkey PRIMARY KEY (id);


--
-- Name: scanner_registration scanner_registration_url_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_registration
    ADD CONSTRAINT scanner_registration_url_key UNIQUE (url);


--
-- Name: scanner_registration scanner_registration_uuid_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scanner_registration
    ADD CONSTRAINT scanner_registration_uuid_key UNIQUE (uuid);


--
-- Name: schedule schedule_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedule
    ADD CONSTRAINT schedule_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: system_artifact system_artifact_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.system_artifact
    ADD CONSTRAINT system_artifact_pkey PRIMARY KEY (id);


--
-- Name: system_artifact system_artifact_repository_digest_vendor_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.system_artifact
    ADD CONSTRAINT system_artifact_repository_digest_vendor_key UNIQUE (repository, digest, vendor);


--
-- Name: tag tag_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag
    ADD CONSTRAINT tag_pkey PRIMARY KEY (id);


--
-- Name: task task_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task
    ADD CONSTRAINT task_pkey PRIMARY KEY (id);


--
-- Name: artifact unique_artifact; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact
    ADD CONSTRAINT unique_artifact UNIQUE (repository_id, digest);


--
-- Name: artifact_accessory unique_artifact_accessory; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_accessory
    ADD CONSTRAINT unique_artifact_accessory UNIQUE (artifact_id, subject_artifact_digest);


--
-- Name: artifact_blob unique_artifact_blob; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_blob
    ADD CONSTRAINT unique_artifact_blob UNIQUE (digest_af, digest_blob);


--
-- Name: artifact_trash unique_artifact_trash; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_trash
    ADD CONSTRAINT unique_artifact_trash UNIQUE (repository_name, digest);


--
-- Name: user_group unique_group_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_group
    ADD CONSTRAINT unique_group_name UNIQUE (group_name);


--
-- Name: harbor_label unique_label; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.harbor_label
    ADD CONSTRAINT unique_label UNIQUE (name, scope, project_id);


--
-- Name: label_reference unique_label_reference; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.label_reference
    ADD CONSTRAINT unique_label_reference UNIQUE (label_id, artifact_id);


--
-- Name: replication_policy unique_policy_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.replication_policy
    ADD CONSTRAINT unique_policy_name UNIQUE (name);


--
-- Name: project_blob unique_project_blob; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_blob
    ADD CONSTRAINT unique_project_blob UNIQUE (project_id, blob_id);


--
-- Name: project_member unique_project_entity_type; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_member
    ADD CONSTRAINT unique_project_entity_type UNIQUE (project_id, entity_id, entity_type);


--
-- Name: project_metadata unique_project_id_and_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_metadata
    ADD CONSTRAINT unique_project_id_and_name UNIQUE (project_id, name);


--
-- Name: permission_policy unique_rbac_policy; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.permission_policy
    ADD CONSTRAINT unique_rbac_policy UNIQUE (scope, resource, action, effect);


--
-- Name: artifact_reference unique_reference; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_reference
    ADD CONSTRAINT unique_reference UNIQUE (parent_id, child_id);


--
-- Name: robot unique_robot; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.robot
    ADD CONSTRAINT unique_robot UNIQUE (name, project_id);


--
-- Name: role_permission unique_role_permission; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.role_permission
    ADD CONSTRAINT unique_role_permission UNIQUE (role_type, role_id, permission_policy_id);


--
-- Name: schedule unique_schedule; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedule
    ADD CONSTRAINT unique_schedule UNIQUE (vendor_type, vendor_id);


--
-- Name: tag unique_tag; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag
    ADD CONSTRAINT unique_tag UNIQUE (repository_id, name);


--
-- Name: registry unique_target_name; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.registry
    ADD CONSTRAINT unique_target_name UNIQUE (name);


--
-- Name: user_group user_group_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_group
    ADD CONSTRAINT user_group_pkey PRIMARY KEY (id);


--
-- Name: vulnerability_record vulnerability_record_cve_id_registration_uuid_package_packa_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vulnerability_record
    ADD CONSTRAINT vulnerability_record_cve_id_registration_uuid_package_packa_key UNIQUE (cve_id, registration_uuid, package, package_version);


--
-- Name: vulnerability_record vulnerability_record_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vulnerability_record
    ADD CONSTRAINT vulnerability_record_pkey PRIMARY KEY (id);


--
-- Name: idx_artifact_accessory_subject_artifact_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifact_accessory_subject_artifact_id ON public.artifact_accessory USING btree (subject_artifact_id);


--
-- Name: idx_artifact_blob_digest_blob; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifact_blob_digest_blob ON public.artifact_blob USING btree (digest_blob);


--
-- Name: idx_artifact_digest_project_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifact_digest_project_id ON public.artifact USING btree (digest, project_id);


--
-- Name: idx_artifact_push_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifact_push_time ON public.artifact USING btree (push_time);


--
-- Name: idx_artifact_reference_child_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifact_reference_child_id ON public.artifact_reference USING btree (child_id);


--
-- Name: idx_artifact_repository_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_artifact_repository_name ON public.artifact USING btree (repository_name);


--
-- Name: idx_audit_log_ext_op_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_audit_log_ext_op_time ON public.audit_log_ext USING btree (op_time);


--
-- Name: idx_audit_log_ext_project_id_operation; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_audit_log_ext_project_id_operation ON public.audit_log_ext USING btree (project_id, operation);


--
-- Name: idx_audit_log_ext_project_id_optime; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_audit_log_ext_project_id_optime ON public.audit_log_ext USING btree (project_id, op_time);


--
-- Name: idx_audit_log_ext_project_id_resource_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_audit_log_ext_project_id_resource_type ON public.audit_log_ext USING btree (project_id, resource_type);


--
-- Name: idx_audit_log_op_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_audit_log_op_time ON public.audit_log USING btree (op_time);


--
-- Name: idx_audit_log_project_id_optime; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_audit_log_project_id_optime ON public.audit_log USING btree (project_id, op_time);


--
-- Name: idx_execution_start_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_execution_start_time ON public.execution USING btree (start_time);


--
-- Name: idx_execution_vendor_type_vendor_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_execution_vendor_type_vendor_id ON public.execution USING btree (vendor_type, vendor_id);


--
-- Name: idx_report_vulnerability_record_vuln_record_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_report_vulnerability_record_vuln_record_id ON public.report_vulnerability_record USING btree (vuln_record_id);


--
-- Name: idx_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_status ON public.blob USING btree (status);


--
-- Name: idx_tag_artifact_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tag_artifact_id ON public.tag USING btree (artifact_id);


--
-- Name: idx_tag_push_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tag_push_time ON public.tag USING btree (push_time);


--
-- Name: idx_task_extra_attrs_report_uuids; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_task_extra_attrs_report_uuids ON public.task USING gin ((((extra_attrs)::jsonb -> 'report_uuids'::text)));


--
-- Name: idx_task_job_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_task_job_id ON public.task USING btree (job_id);


--
-- Name: idx_version; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_version ON public.blob USING btree (version);


--
-- Name: idx_vulnerability_record_cve_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vulnerability_record_cve_id ON public.vulnerability_record USING btree (cve_id);


--
-- Name: idx_vulnerability_record_cvss_score_v3; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vulnerability_record_cvss_score_v3 ON public.vulnerability_record USING btree (cvss_score_v3);


--
-- Name: idx_vulnerability_record_package; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vulnerability_record_package ON public.vulnerability_record USING btree (package);


--
-- Name: idx_vulnerability_record_severity; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vulnerability_record_severity ON public.vulnerability_record USING btree (severity);


--
-- Name: idx_vulnerability_registration_uuid; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vulnerability_registration_uuid ON public.vulnerability_record USING btree (registration_uuid);


--
-- Name: job_log_uuid; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX job_log_uuid ON public.job_log USING btree (job_uuid);


--
-- Name: task_execution_id_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX task_execution_id_idx ON public.task USING btree (execution_id);


--
-- Name: harbor_label harbor_label_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER harbor_label_update_time_at_modtime BEFORE UPDATE ON public.harbor_label FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: harbor_user harbor_user_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER harbor_user_update_time_at_modtime BEFORE UPDATE ON public.harbor_user FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: oidc_user oidc_user_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER oidc_user_update_time_at_modtime BEFORE UPDATE ON public.oidc_user FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: project_member project_member_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER project_member_update_time_at_modtime BEFORE UPDATE ON public.project_member FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: project_metadata project_metadata_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER project_metadata_update_time_at_modtime BEFORE UPDATE ON public.project_metadata FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: project project_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER project_update_time_at_modtime BEFORE UPDATE ON public.project FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: replication_policy replication_policy_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER replication_policy_update_time_at_modtime BEFORE UPDATE ON public.replication_policy FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: repository repository_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER repository_update_time_at_modtime BEFORE UPDATE ON public.repository FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: robot robot_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER robot_update_time_at_modtime BEFORE UPDATE ON public.robot FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: user_group user_group_update_time_at_modtime; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER user_group_update_time_at_modtime BEFORE UPDATE ON public.user_group FOR EACH ROW EXECUTE FUNCTION public.update_update_time_at_column();


--
-- Name: artifact_accessory artifact_accessory_artifact_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_accessory
    ADD CONSTRAINT artifact_accessory_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES public.artifact(id);


--
-- Name: artifact_reference artifact_reference_child_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_reference
    ADD CONSTRAINT artifact_reference_child_id_fkey FOREIGN KEY (child_id) REFERENCES public.artifact(id);


--
-- Name: artifact_reference artifact_reference_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.artifact_reference
    ADD CONSTRAINT artifact_reference_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.artifact(id);


--
-- Name: vulnerability_record fk_registration_uuid; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vulnerability_record
    ADD CONSTRAINT fk_registration_uuid FOREIGN KEY (registration_uuid) REFERENCES public.scanner_registration(uuid) ON DELETE CASCADE;


--
-- Name: report_vulnerability_record fk_report_uuid; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report_vulnerability_record
    ADD CONSTRAINT fk_report_uuid FOREIGN KEY (report_uuid) REFERENCES public.scan_report(uuid) ON DELETE CASCADE;


--
-- Name: report_vulnerability_record fk_vuln_record_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report_vulnerability_record
    ADD CONSTRAINT fk_vuln_record_id FOREIGN KEY (vuln_record_id) REFERENCES public.vulnerability_record(id) ON DELETE CASCADE;


--
-- Name: label_reference label_reference_artifact_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.label_reference
    ADD CONSTRAINT label_reference_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES public.artifact(id);


--
-- Name: label_reference label_reference_label_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.label_reference
    ADD CONSTRAINT label_reference_label_id_fkey FOREIGN KEY (label_id) REFERENCES public.harbor_label(id);


--
-- Name: oidc_user oidc_user_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.oidc_user
    ADD CONSTRAINT oidc_user_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.harbor_user(user_id);


--
-- Name: project_metadata project_metadata_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project_metadata
    ADD CONSTRAINT project_metadata_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.project(project_id);


--
-- Name: project project_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.project
    ADD CONSTRAINT project_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.harbor_user(user_id);


--
-- Name: tag tag_artifact_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag
    ADD CONSTRAINT tag_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES public.artifact(id);


--
-- Name: task task_execution_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.task
    ADD CONSTRAINT task_execution_id_fkey FOREIGN KEY (execution_id) REFERENCES public.execution(id);


--
-- PostgreSQL database dump complete
--

\unrestrict 7AzeAKyl9UFeIRlbHi8zdDU2nagbT4cT9Lcvy2g7zdd0pY7X2aTG7lnbeTnq8Yk

